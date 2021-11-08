export namespace symbols {
    export const Arithmetics = Symbol.for('arithmetics')
}

export interface InstanceOf<A> {
    [symbols.Arithmetics]: A
    clone(): InstanceOf<A>
}
