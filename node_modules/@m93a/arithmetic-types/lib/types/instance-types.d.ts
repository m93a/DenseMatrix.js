export declare namespace symbols {
    const Arithmetics: unique symbol;
}
export interface InstanceOf<A> {
    [symbols.Arithmetics]: A;
    clone(): InstanceOf<A>;
}
