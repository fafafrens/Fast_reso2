"""
    RowBandMatrix(matrix)

Store the contiguous nonzero span of every matrix row. `column_starts[row]`
contains the first source column, while `value_offsets[row]` and
`value_offsets[row + 1]` delimit its values in the flat payload. Zeros inside
a span are retained; numerical zeros before and after it are omitted. Signed
zeros and non-finite values are retained so converting the packed
representation back to a dense matrix is exact.

Compact integer types are selected independently for column and value
positions. Pass them explicitly as `RowBandMatrix(matrix, UInt32, UInt32)`
when a fixed representation is needed.
"""
struct RowBandMatrix{
        T <: Number,
        I <: Integer,
        O <: Integer,
    } <: AbstractMatrix{T}
    column_count::Int
    column_starts::Vector{I}
    value_offsets::Vector{O}
    values::Vector{T}

    function RowBandMatrix(
            column_count::Int,
            column_starts::Vector{I},
            value_offsets::Vector{O},
            values::Vector{T},
        ) where {T <: Number, I <: Integer, O <: Integer}
        _validate_rowband_storage(
            column_count,
            column_starts,
            value_offsets,
            values,
        )
        return new{T, I, O}(
            column_count,
            column_starts,
            value_offsets,
            values,
        )
    end
end

function _validate_rowband_storage(
        column_count,
        column_starts::Vector{I},
        value_offsets::Vector{O},
        values,
    ) where {I <: Integer, O <: Integer}
    if column_count < 0
        throw(ArgumentError("column count must be nonnegative"))
    end
    if column_count > typemax(I)
        throw(
            ArgumentError(
                "$I cannot represent $column_count matrix columns",
            )
        )
    end
    if length(value_offsets) != length(column_starts) + 1
        throw(
            ArgumentError(
                "value offsets need one entry per row plus a sentinel",
            )
        )
    end
    if length(values) == typemax(Int) ||
            length(values) + 1 > typemax(O)
        throw(
            ArgumentError(
                "$O cannot represent $(length(values)) stored values",
            )
        )
    end

    final_offset = O(length(values) + 1)
    if first(value_offsets) != one(O) ||
            last(value_offsets) != final_offset
        throw(
            ArgumentError(
                "value offsets must cover the complete flat payload",
            )
        )
    end

    previous_offset = one(O)
    for offset in value_offsets
        if !(one(O) <= offset <= final_offset)
            throw(
                ArgumentError(
                    "value offset $offset is outside the payload",
                )
            )
        end
        if offset < previous_offset
            throw(ArgumentError("value offsets must be nondecreasing"))
        end
        previous_offset = offset
    end

    last_column = I(column_count)
    for row in eachindex(column_starts)
        first_value = Int(value_offsets[row])
        next_value = Int(value_offsets[row + 1])
        band_length = next_value - first_value
        stored_column = column_starts[row]

        if band_length == 0
            if stored_column != one(I)
                throw(
                    ArgumentError(
                        "an empty row must use column start 1",
                    )
                )
            end
        elseif !(one(I) <= stored_column <= last_column)
            throw(
                ArgumentError(
                    "row $row starts outside columns 1:$column_count",
                )
            )
        else
            first_column = Int(stored_column)
            if band_length > column_count - first_column + 1
                throw(
                    ArgumentError(
                        "row $row band extends past column $column_count",
                    )
                )
            end
        end
    end
    return nothing
end

Base.size(matrix::RowBandMatrix) =
    (length(matrix.column_starts), matrix.column_count)

Base.IndexStyle(::Type{<:RowBandMatrix}) = IndexCartesian()

function _rowbandcolumntype(column_count)
    if column_count ≤ typemax(UInt8)
        return UInt8
    elseif column_count ≤ typemax(UInt16)
        return UInt16
    elseif column_count ≤ typemax(UInt32)
        return UInt32
    end
    return Int
end

function _rowbandoffsettype(value_count)
    if value_count < typemax(UInt16)
        return UInt16
    elseif value_count < typemax(UInt32)
        return UInt32
    end
    return Int
end

@inline function _rowbandlength(matrix::RowBandMatrix, row::Int)
    return Int(matrix.value_offsets[row + 1]) -
        Int(matrix.value_offsets[row])
end

@inline function _rowband(matrix::RowBandMatrix, row::Int)
    first_column = Int(matrix.column_starts[row])
    return first_column:(first_column + _rowbandlength(matrix, row) - 1)
end

Base.@propagate_inbounds function Base.getindex(
        matrix::RowBandMatrix{T},
        row::Int,
        column::Int,
    ) where {T}
    @boundscheck checkbounds(matrix, row, column)

    @inbounds begin
        first_column = Int(matrix.column_starts[row])
        column_offset = column - first_column
        band_length = _rowbandlength(matrix, row)
        if 0 <= column_offset < band_length
            value_index =
                Int(matrix.value_offsets[row]) + column_offset
            return matrix.values[value_index]
        end
    end
    return zero(T)
end

function _check_rowband_growth(
        matrix::RowBandMatrix{T, I, O},
        added_values::Int,
    ) where {T, I, O}
    stored_values = length(matrix.values)
    if added_values > typemax(Int) - stored_values
        throw(ArgumentError("row-band payload exceeds typemax(Int)"))
    end

    new_value_count = stored_values + added_values
    if new_value_count == typemax(Int) ||
            new_value_count + 1 > typemax(O)
        throw(
            ArgumentError(
                "$O cannot represent $new_value_count stored values",
            )
        )
    end
    return new_value_count
end

function _growrowband!(
        matrix::RowBandMatrix{T, I, O},
        row::Int,
        first_value::Int,
        next_value::Int,
        prefix_length::Int,
        suffix_length::Int,
    ) where {T, I, O}
    added_values = prefix_length + suffix_length
    new_value_count = _check_rowband_growth(matrix, added_values)
    old_value_count = length(matrix.values)
    zero_value = zero(T)
    resize!(matrix.values, new_value_count)

    @inbounds begin
        tail_length = old_value_count - next_value + 1
        if tail_length > 0
            copyto!(
                matrix.values,
                next_value + added_values,
                matrix.values,
                next_value,
                tail_length,
            )
        end
        row_length = next_value - first_value
        if prefix_length > 0 && row_length > 0
            copyto!(
                matrix.values,
                first_value + prefix_length,
                matrix.values,
                first_value,
                row_length,
            )
        end
        for value_index in first_value:(first_value + prefix_length - 1)
            matrix.values[value_index] = zero_value
        end
        suffix_start = next_value + prefix_length
        for value_index in suffix_start:(suffix_start + suffix_length - 1)
            matrix.values[value_index] = zero_value
        end

        for offset_index in (row + 1):length(matrix.value_offsets)
            matrix.value_offsets[offset_index] = O(
                Int(matrix.value_offsets[offset_index]) + added_values,
            )
        end
    end
    return matrix
end

Base.@propagate_inbounds function Base.setindex!(
        matrix::RowBandMatrix{T, I, O},
        value,
        row::Int,
        column::Int,
    ) where {T, I, O}
    @boundscheck checkbounds(matrix, row, column)
    stored_value = convert(T, value)

    @inbounds begin
        first_value = Int(matrix.value_offsets[row])
        next_value = Int(matrix.value_offsets[row + 1])
        band_length = next_value - first_value
        first_column = Int(matrix.column_starts[row])
        column_offset = column - first_column

        if 0 <= column_offset < band_length
            matrix.values[first_value + column_offset] = stored_value
            return matrix
        end
        if isequal(stored_value, zero(T))
            return matrix
        end

        if band_length == 0
            _growrowband!(
                matrix,
                row,
                first_value,
                next_value,
                0,
                1,
            )
            matrix.column_starts[row] = I(column)
            matrix.values[first_value] = stored_value
            return matrix
        end

        last_column = first_column + band_length - 1
        new_first_column = min(first_column, column)
        prefix_length = first_column - new_first_column
        suffix_length = max(column - last_column, 0)
        _growrowband!(
            matrix,
            row,
            first_value,
            next_value,
            prefix_length,
            suffix_length,
        )
        matrix.column_starts[row] = I(new_first_column)
        matrix.values[first_value + column - new_first_column] =
            stored_value
    end
    return matrix
end

function Base.fill!(matrix::RowBandMatrix{T, I, O}, value) where {T, I, O}
    stored_value = convert(T, value)
    if isequal(stored_value, zero(T))
        fill!(matrix.values, stored_value)
        return matrix
    end

    row_count, column_count = size(matrix)
    full_value_count = Base.checked_mul(row_count, column_count)
    added_values = full_value_count - length(matrix.values)
    _check_rowband_growth(matrix, added_values)

    resize!(matrix.values, full_value_count)
    fill!(matrix.values, stored_value)
    fill!(matrix.column_starts, one(I))
    @inbounds for row in 0:row_count
        matrix.value_offsets[row + 1] = O(row * column_count + 1)
    end
    return matrix
end

RowBandMatrix(matrix::AbstractMatrix{T}) where {T <: Number} =
    RowBandMatrix(matrix, _rowbandcolumntype(size(matrix, 2)))

function RowBandMatrix(
        matrix::AbstractMatrix{T},
        ::Type{I},
    ) where {T <: Number, I <: Integer}
    column_starts, band_lengths, value_count =
        _rowbandmetadata(matrix, I)
    return _packrowbands(
        matrix,
        column_starts,
        band_lengths,
        value_count,
        _rowbandoffsettype(length(matrix)),
    )
end

function RowBandMatrix(
        matrix::AbstractMatrix{T},
        ::Type{I},
        ::Type{O},
    ) where {T <: Number, I <: Integer, O <: Integer}
    column_starts, band_lengths, value_count =
        _rowbandmetadata(matrix, I)
    return _packrowbands(
        matrix,
        column_starts,
        band_lengths,
        value_count,
        O,
    )
end

function _rowbandmetadata(
        matrix::AbstractMatrix,
        ::Type{I},
    ) where {I <: Integer}
    Base.require_one_based_indexing(matrix)
    row_count, column_count = size(matrix)
    if column_count > typemax(I)
        throw(
            ArgumentError(
                "$I cannot represent $column_count matrix columns",
            )
        )
    end

    column_starts = Vector{I}(undef, row_count)
    band_lengths = Vector{Int}(undef, row_count)
    value_count = 0
    zero_value = zero(eltype(matrix))

    @inbounds for row in 1:row_count
        first_column = 0
        last_column = 0
        for column in 1:column_count
            if !isequal(matrix[row, column], zero_value)
                if first_column == 0
                    first_column = column
                end
                last_column = column
            end
        end

        if last_column == 0
            column_starts[row] = one(I)
            band_lengths[row] = 0
        else
            band_length = last_column - first_column + 1
            column_starts[row] = I(first_column)
            band_lengths[row] = band_length
            value_count += band_length
        end
    end
    return column_starts, band_lengths, value_count
end

function _packrowbands(
        matrix::AbstractMatrix{T},
        column_starts::Vector{I},
        band_lengths::Vector{Int},
        value_count::Int,
        ::Type{O},
    ) where {T <: Number, I <: Integer, O <: Integer}
    if value_count == typemax(Int) ||
            value_count + 1 > typemax(O)
        throw(
            ArgumentError(
                "$O cannot represent $value_count stored values",
            )
        )
    end

    value_offsets = Vector{O}(undef, length(column_starts) + 1)
    values = Vector{T}(undef, value_count)
    value_offsets[1] = one(O)
    value_index = 1
    @inbounds for row in eachindex(column_starts)
        first_column = Int(column_starts[row])
        for column_offset in 0:(band_lengths[row] - 1)
            column = first_column + column_offset
            values[value_index] = matrix[row, column]
            value_index += 1
        end
        value_offsets[row + 1] = O(value_index)
    end
    return RowBandMatrix(
        size(matrix, 2),
        column_starts,
        value_offsets,
        values,
    )
end

function Base.Matrix(matrix::RowBandMatrix{T}) where {T}
    dense = zeros(T, size(matrix))
    @inbounds for row in eachindex(matrix.column_starts)
        first_value = Int(matrix.value_offsets[row])
        next_value = Int(matrix.value_offsets[row + 1])
        first_column = Int(matrix.column_starts[row])
        for value_index in first_value:(next_value - 1)
            column = first_column + value_index - first_value
            dense[row, column] = matrix.values[value_index]
        end
    end
    return dense
end

Base.copy(matrix::RowBandMatrix) = RowBandMatrix(
    matrix.column_count,
    copy(matrix.column_starts),
    copy(matrix.value_offsets),
    copy(matrix.values),
)

function Base.similar(
        matrix::RowBandMatrix{<:Number, I, O},
        ::Type{T},
        dimensions::Dims{2},
    ) where {T <: Number, I <: Integer, O <: Integer}
    if dimensions == size(matrix)
        return RowBandMatrix(
            matrix.column_count,
            copy(matrix.column_starts),
            copy(matrix.value_offsets),
            Vector{T}(undef, length(matrix.values)),
        )
    end

    row_count, column_count = dimensions
    column_type = _rowbandcolumntype(column_count)
    offset_type = _rowbandoffsettype(
        Base.checked_mul(row_count, column_count),
    )
    return RowBandMatrix(
        column_count,
        fill(one(column_type), row_count),
        fill(one(offset_type), row_count + 1),
        Vector{T}(undef, 0),
    )
end

Base.similar(
    ::RowBandMatrix,
    ::Type{T},
    dimensions::Dims,
) where {T} = Array{T}(undef, dimensions)
