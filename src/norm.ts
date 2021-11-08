import { symbols, NormedDivisionRing, Real, Ring } from "@m93a/arithmetic-types"
import { DenseMatrix, InstanceOf } from "./DenseMatrix"
import { ArithmeticError } from "./error/ArithmeticError"
import { DimensionError } from "./error/DimensionError"
import { NotImplementedError } from "./error/NotImplementedError"

export type pValue = number | '-inf' | 'inf' | 'fro'

/**
 * Calculate the norm for a vector or matrix
 */
export function norm<
    ScalarArithmetics extends Ring<any>
>(
    M: DenseMatrix<ScalarArithmetics>,
    p?: number | '-inf' | 'inf' | 'fro'
): InstanceOf<ScalarArithmetics> {

    type Scalar = InstanceOf<ScalarArithmetics>

    // Frobenius norm == sum of squares of entries
    if (p === 'fro' || p === undefined) {
        const A = M.scalarArithmetics as any as NormedDivisionRing<Scalar, any>
        if (A[symbols.NormedDivisionRing] !== true) throw new ArithmeticError('NormedDivisionRing')

        let r = A.zero()
        for (const { value } of M) {
            r = A.add(r, A.pow(A.norm(value), 2))
        }

        return A.pow(r, 1/2)
    }

    // Other special norms
    switch (M.size.length) {
        case 1:
            return vectorNorm<Scalar>(M, p)

        case 2:
            return matrixNorm<Scalar>(M, p)

        default:
            throw new DimensionError(M.size.length, 2, '>')
    }
}


function vectorNorm<Scalar>(v: DenseMatrix<any, any>, p: number | '-inf' | 'inf'): Scalar {

    switch (p) {

        // Maximum norm
        case Number.POSITIVE_INFINITY:
        case 'inf': {
            const A = v.scalarArithmetics as Real<Scalar>
            if (A[symbols.Real] !== true) throw new ArithmeticError('Real')

            let r = A.zero()
            for (const { value } of v) {
                const x = A.norm(value)
                if (A.gt(x, r)) {
                    r = x
                }
            }
            return r
        }

        // Minimum absolute value (pseudo)norm
        case Number.NEGATIVE_INFINITY:
        case '-inf': {
            const A = v.scalarArithmetics as Real<Scalar>
            if (A[symbols.Real] !== true) throw new ArithmeticError('Real')

            let r = A.zero()
            for (const { value } of v) {
                const x = A.norm(value)
                if (A.lt(x, r)) {
                    r = x
                }
            }
            return r
        }

        // L^p norm
        default: {
            const A = v.scalarArithmetics as NormedDivisionRing<Scalar, any>
            if (A[symbols.NormedDivisionRing] !== true) throw new ArithmeticError('NormedDivisionRing')

            if (p === 0) return A.fromNumber(Number.POSITIVE_INFINITY)

            let r = A.zero()
            for (const { value } of v) {
                r = A.add(r, A.pow(A.norm(value), p))
            }

            return A.pow(r, 1/p)
        }
    }
}


function matrixNorm<Scalar>(M: DenseMatrix<any, any>, p: number | '-inf' | 'inf'): Scalar {
    switch (p) {
        // Largest column maximum norm
        case 1: {
            const A = M.scalarArithmetics as Real<Scalar>
            if (A[symbols.Real] !== true) throw new ArithmeticError('Real')

            let r = A.zero()
            for (const col of M.columns() as any as Scalar[][]) {
                const colSum = col.map(A.norm).reduce(A.add)
                if (A.gt(colSum, r)) r = colSum
            }
            return r
        }

        // Largest row maximum norm
        case Number.POSITIVE_INFINITY:
        case 'inf': {
            const A = M.scalarArithmetics as Real<Scalar>
            if (A[symbols.Real] !== true) throw new ArithmeticError('Real')

            let r = A.zero()
            for (const row of M.rows() as any as Scalar[][]) {
                const rowSum = row.map(A.norm).reduce(A.add)
                if (A.gt(rowSum, r)) r = rowSum
            }
            return r
        }

        case '-inf': {
            throw new TypeError('L^-inf norm is not defined for square matrices')
        }

        // L^p norm
        default: {
            // TODO, requires eigs
            throw new NotImplementedError('L^p norm for square matrices')
        }
    }
}
