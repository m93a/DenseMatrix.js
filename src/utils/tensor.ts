import { symbols, Tensor, InstanceOf } from '@m93a/arithmetic-types'

type TensorInstance<T, F> = InstanceOf<Tensor<T, F>>
type MultidimArray<T> = T[] | MultidimArray<T>[]

export function forEach<F, T extends TensorInstance<T, F>>(
    tensor: T,
    callback: (value: F, index: number[]) => void
) {
    const arithmetics = tensor[symbols.Arithmetics]
    const size = arithmetics.size(tensor)

    const indices = Array(size.length).fill(0)

    loopElements:
    while (true) {
        callback(arithmetics.get(tensor, indices), indices)

        incrementIndices:
        for (let i = 0 ;; i++) {
            indices[i]++

            if (i >= size.length) break loopElements
            if (indices[i] < size[i]) break incrementIndices

            indices[i] = 0
        }
    }
}

export function mapToMultidimArray<F, T extends TensorInstance<T, F>, R>(
    tensor: T,
    callback: (value: F, index: number[]) => R
): MultidimArray<R> {
    const arithmetics = tensor[symbols.Arithmetics]
    const size = arithmetics.size(tensor)

    const result: MultidimArray<R> = []

    let topArr: R[]
    let previousIndex: number[]

    forEach<F, T>(tensor, (value, index) => {
        const lastI = index[index.length - 1]

        if (!equalUpToLastElement(index, previousIndex)) topArr = retrieveTopArrayByIndex(result, index)
        previousIndex = [...index]

        topArr[lastI] = callback(value, index)
    })

    return result
}

export function equalUpToLastElement(indexA: number[], indexB: number[]): boolean {
    if (!Array.isArray(indexA) || !Array.isArray(indexB)) return false
    if (indexA.length !== indexB.length) return false

    for (let i = 0; i < indexA.length - 1; i++) {
        if (indexA[i] !== indexB[i]) return false
    }

    return true
}

export function retrieveTopArrayByIndex<T>(multidimArr: MultidimArray<T>, index:number[]): T[] {
    let arr: MultidimArray<T> = multidimArr

    index = [...index]
    index.pop()

    for (const i of index) {
        if (arr[i] === undefined) arr[i] = []
        arr = arr[i] as MultidimArray<T>
    }

    return arr as T[]
}
