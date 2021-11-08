import { symbols as symbols_instanceTypes } from './instance-types'
import { symbols as symbols_scalarArithmetic } from './scalar-arithmetic'
import { symbols as symbols_tensorArithmetic } from './tensor-arithmetic'

export const symbols = {
    ...symbols_instanceTypes,
    ...symbols_scalarArithmetic,
    ...symbols_tensorArithmetic
}

export type symbols = typeof symbols

export * from './instance-types'
export * from './scalar-arithmetic'
export * from './tensor-arithmetic'
export * from './is'
