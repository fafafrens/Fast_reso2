"""
    RowBandMatrix(matrix)

Store the contiguous nonzero span of every matrix row. `column_starts[row]`
contains the first source column, while `value_offsets[row]` and
`value_offsets[row + 1]` delimit its values in the flat payload. Zeros inside
a span are retained; numerical zeros before and after it are omitted. Signed
zeros and non-finite values are retained so converting the packed
representation back to a dense matrix is exact.

The public array interface is read-only and does not support element
assignment. Construct it from a completed dense matrix, then use it for
indexing, conversion, or matrix-vector products.

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

Base.similar(
    ::RowBandMatrix,
    ::Type{T},
    dimensions::Dims,
) where {T} = Array{T}(undef, dimensions)
