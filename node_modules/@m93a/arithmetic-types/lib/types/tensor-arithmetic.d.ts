import { NormedVectorSpace } from "./scalar-arithmetic";
export declare type TensorNormType = number | '-inf' | 'inf' | 'fro';
export declare type ArrayTensor<F> = F[] | ArrayTensor<F>[];
export declare namespace symbols {
    const Tensor: unique symbol;
}
export interface Tensor<T, F> extends NormedVectorSpace<T, F> {
    [symbols.Tensor]: true;
    size(a: T): number[];
    /** Returns the element at the specified index. */
    get(a: T, index: number[]): F;
    /**
     * Returns a new tensor with the element at the specified
     * index replaced by the specified value.
     */
    set?(a: T, index: number[], value: F): T;
    norm(a: T, which?: TensorNormType): F;
    normSq(a: T, which?: TensorNormType): F;
    dotAdd(a: T, b: F): T;
    dotSub(a: T, b: F): T;
    dotMul(a: T, b: F): T;
    dotDiv(a: T, b: F): T;
    dotPow(a: T, b: F): T;
    dotExp(a: T): T;
    dotExpm1(a: T): T;
    kron(a: T, b: T): T;
    zero(): never;
    zero(size: number[]): T;
    fromArray(arr: ArrayTensor<F>): T;
    fromFunction(fn: (index: number[]) => F): T;
    map(a: T, fn: (v: F, index: number[]) => F): T;
    forEach(a: T, fn: (v: F, index: number[]) => void): void;
}
