@inline function _mul_dimensions(destination, matrix, source)
    row_count, column_count = size(matrix)
    if length(destination) != row_count
        throw(
            DimensionMismatch(
                "matrix has $row_count rows but destination has " *
                    "length $(length(destination))",
            )
        )
    end
    if length(source) != column_count
        throw(
            DimensionMismatch(
                "matrix has $column_count columns but source has " *
                    "length $(length(source))",
            )
        )
    end
    return row_count, column_count
end

function mul!(
        destination::AbstractVector{DestinationValue},
        matrix::RowBandMatrix{MatrixValue},
        source::AbstractVector{SourceValue},
        α::Number,
        β::Number,
    ) where {
        DestinationValue <: Number,
        MatrixValue <: Number,
        SourceValue <: Number,
    }
    Base.require_one_based_indexing(destination, matrix, source)
    row_count, column_count =
        _mul_dimensions(destination, matrix, source)

    if iszero(α) || iszero(column_count)
        if iszero(β)
            fill!(destination, zero(DestinationValue))
        elseif !isone(β)
            @inbounds @simd for row in eachindex(destination)
                destination[row] *= β
            end
        end
        return destination
    end

    product_type = Base.promote_op(*, MatrixValue, SourceValue)
    accumulator_type = Base.promote_op(+, product_type, product_type)

    if iszero(β)
        @inbounds for row in 1:row_count
            first_value = Int(matrix.value_offsets[row])
            next_value = Int(matrix.value_offsets[row + 1])
            first_column = Int(matrix.column_starts[row])
            accumulator = zero(accumulator_type)
            @simd for value_index in first_value:(next_value - 1)
                accumulator = muladd(
                    matrix.values[value_index],
                    source[first_column + value_index - first_value],
                    accumulator,
                )
            end
            destination[row] = α * accumulator
        end
    else
        if !isone(β)
            @inbounds @simd for row in eachindex(destination)
                destination[row] *= β
            end
        end

        @inbounds for row in 1:row_count
            first_value = Int(matrix.value_offsets[row])
            next_value = Int(matrix.value_offsets[row + 1])
            first_column = Int(matrix.column_starts[row])
            accumulator = zero(accumulator_type)
            @simd for value_index in first_value:(next_value - 1)
                accumulator = muladd(
                    matrix.values[value_index],
                    source[first_column + value_index - first_value],
                    accumulator,
                )
            end
            destination[row] = muladd(α, accumulator, destination[row])
        end
    end
    return destination
end

@inline _cascade_axpy!(α, source, destination) =
    axpy!(α, source, destination)

# Avoid fixed BLAS call overhead for the plan's short scatters. Real weights
# scale complex spectra component-wise, so both spectrum types share the loop.
function _cascade_axpy!(
        α::Float64,
        source::Vector{S},
        destination::Vector{S},
    ) where {S <: Union{Float64, ComplexF64}}
    iszero(α) && return destination

    @inbounds @simd for index in eachindex(source, destination)
        destination[index] =
            muladd(α, source[index], destination[index])
    end
    return destination
end
