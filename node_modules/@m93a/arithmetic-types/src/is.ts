import { AdditiveGroup, DivisionRing, NormedDivisionRing, NormedVectorSpace, Real, Ring, UnitarySpace, VectorSpace } from './scalar-arithmetic'
import { Tensor } from './tensor-arithmetic'
import { InstanceOf } from './instance-types'
import { symbols } from '.'

export function isAdditiveGroup(x: any): x is InstanceOf<AdditiveGroup<any>> {
    return x?.[symbols.Arithmetics]?.[symbols.AdditiveGroup]
}

export function isDivisionRing(x: any): x is InstanceOf<DivisionRing<any>> {
    return x?.[symbols.Arithmetics]?.[symbols.DivisionRing]
}

export function isNormedDivisionRing(x: any): x is InstanceOf<NormedDivisionRing<any, any>> {
    return x?.[symbols.Arithmetics]?.[symbols.NormedDivisionRing]
}

export function isNormedVectorSpace(x: any): x is InstanceOf<NormedVectorSpace<any, any>> {
    return x?.[symbols.Arithmetics]?.[symbols.NormedVectorSpace]
}

export function isReal(x: any): x is InstanceOf<Real<any>> {
    return x?.[symbols.Arithmetics]?.[symbols.Real]
}

export function isRing(x: any): x is InstanceOf<Ring<any>> {
    return x?.[symbols.Arithmetics]?.[symbols.Ring]
}

export function isUnitarySpace(x: any): x is InstanceOf<UnitarySpace<any, any>> {
    return x?.[symbols.Arithmetics]?.[symbols.UnitarySpace]
}

export function isVectorSpace(x: any): x is InstanceOf<VectorSpace<any, any>> {
    return x?.[symbols.Arithmetics]?.[symbols.VectorSpace]
}

export function isTensor(x: any): x is InstanceOf<Tensor<any, any>> {
    return x?.[symbols.Arithmetics]?.[symbols.Tensor]
}
