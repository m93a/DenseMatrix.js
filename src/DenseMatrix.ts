import { symbols, Ring, Tensor, InstanceOf as _InstanceOf, isTensor, DivisionRing, NormedDivisionRing } from '@m93a/arithmetic-types'
import { arraySize, getArrayDataType, MultidimArray, resize, validateIndex } from './utils/array'
import { clone } from './utils/object'
import { mapToMultidimArray, retrieveTopArrayByIndex } from './utils/tensor'
import { isArray, typeOf } from './utils/is'
import { DimensionError } from './error/DimensionError'
import { format } from './utils/string'
import { NumberArithmetics } from './utils/numberArithmetic'
import { norm } from './norm'
import { ArithmeticError } from './error/ArithmeticError'
import { InternalError } from './error/InternalError'

export type InstanceOf<T> = T extends Ring<number> ? number : _InstanceOf<T>
type TensorInstance<T, R> = _InstanceOf<Tensor<T, R>>

export function isDenseMatrix(x: any): x is DenseMatrix<any, any> {
    return typeof x === 'object' && x[symbols.Tensor] && x.type === 'DenseMatrix'
}

interface SerializedDenseMatrix<
    ScalarArithmetics extends Ring<any> = Ring<any>,
    Scalar extends InstanceOf<ScalarArithmetics> = InstanceOf<ScalarArithmetics>
> {
    mathjs: 'DenseMatrix'
    data: MultidimArray<Scalar>
    size: number[]
    datatype?: any
}

export function isSerializedDenseMatrix(x: any): x is SerializedDenseMatrix {
    return typeof x === 'object' && x.mathjs === 'DenseMatrix' && isArray(x.data) && isArray(x.size)
}

/**
 * Preprocess data, which can be an Array or DenseMatrix with nested Arrays and
 * Matrices. Replaces all nested Matrices with Arrays
 * @memberof DenseMatrix
 * @param {Array} data
 * @return {Array} data
 */
function preprocess<
    ScalarArithmetics extends Ring<any> = Ring<number>,
    Scalar extends InstanceOf<ScalarArithmetics> = InstanceOf<ScalarArithmetics>
>(
    data: MultidimArray<Scalar | DenseMatrix<ScalarArithmetics, Scalar>>
): MultidimArray<Scalar>
{
    for (let i = 0, ii = data.length; i < ii; i++) {
        const elem = data[i]
        if (isArray(elem)) {
            data[i] = preprocess<ScalarArithmetics, Scalar>(elem)
        } else if (elem && isDenseMatrix(elem)) {
            data[i] = preprocess<ScalarArithmetics, Scalar>(elem.valueOf())
        }
    }

    return data as MultidimArray<Scalar>
}





/**
 * Dense Matrix implementation. A regular, dense matrix, supporting multi-dimensional matrices. This is the default matrix type.
 * @enum {{ value, index: number[] }}
 */
export class DenseMatrix<
    ScalarArithmetics extends Ring<any> = typeof NumberArithmetics,
    Scalar extends InstanceOf<ScalarArithmetics> = InstanceOf<ScalarArithmetics>
>
implements TensorInstance<DenseMatrix<ScalarArithmetics>, ScalarArithmetics>
{

    // #region Constructor, basic props
    /*
     *    _____                _                   _
     *   / ____|              | |                 | |
     *  | |     ___  _ __  ___| |_ _ __ _   _  ___| |_ ___  _ __
     *  | |    / _ \| '_ \/ __| __| '__| | | |/ __| __/ _ \| '__|
     *  | |___| (_) | | | \__ \ |_| |  | |_| | (__| || (_) | |
     *   \_____\___/|_| |_|___/\__|_|   \__,_|\___|\__\___/|_|
     */

    public static [symbols.Tensor] = true as const
    public static [symbols.NormedVectorSpace] = true as const
    public static [symbols.VectorSpace] = true as const
    public static [symbols.AdditiveGroup] = true as const

    public readonly type = 'DenseMatrix'
    public readonly isDenseMatrix = true

    public [symbols.Arithmetics] = DenseMatrix as any
    public scalarArithmetics: ScalarArithmetics;

    #data: MultidimArray<Scalar>
    #size: number[]
    #datatype: any

    get size(): number[] { return [...this.#size] }
    static size<ScalarArithmetics extends Ring<any>>(m: DenseMatrix<ScalarArithmetics>) { return m.size }

    get datatype() {
        this.#datatype = this.#datatype ?? getArrayDataType(this.#data, typeOf)
        return this.#datatype
    }


    constructor(
        data: MultidimArray<Scalar> |
        TensorInstance<any, ScalarArithmetics> |
        DenseMatrix<ScalarArithmetics, Scalar> |
        SerializedDenseMatrix<ScalarArithmetics, Scalar>,
        datatype?: any
    ) {
        if (isArray(data)) {
            this.#data = preprocess<ScalarArithmetics, Scalar>(data)
            this.#size = arraySize(this.#data)
        }
        else if (isDenseMatrix(data)) {
            this.#data = clone(data.#data)
            this.#size = [...data.size]
            this.#datatype = datatype ?? data.#datatype
        }
        else if (isSerializedDenseMatrix(data)) {
            this.#data = clone(data.data)
            this.#size = [...data.size]
            this.#datatype = datatype ?? data.datatype
        }
        else if (isTensor(data)) {
            const arithmetics = data[symbols.Arithmetics]
            const size = [...arithmetics.size(data)]
            this.#size = size

            this.#data = mapToMultidimArray(data, (v: any) => clone(v))
        }
        else {
            throw new TypeError('Unsupported type of data (' + typeOf(data) + ')')
        }

        // Find scalar arithmetic
        let el = this.#data[0];
        while (Array.isArray(el)) el = el[0]
        if (typeof el === 'number') this.scalarArithmetics = NumberArithmetics as any
        else this.scalarArithmetics = el[symbols.Arithmetics] as any
    }


    clone(this: DenseMatrix<ScalarArithmetics>): any {
        return new DenseMatrix<ScalarArithmetics>(this)
    }
    // #endregion


    // #region Element access
    /*
     *   ______ _                           _
     *  |  ____| |                         | |
     *  | |__  | | ___ _ __ ___   ___ _ __ | |_    __ _  ___ ___ ___  ___ ___
     *  |  __| | |/ _ \ '_ ` _ \ / _ \ '_ \| __|  / _` |/ __/ __/ _ \/ __/ __|
     *  | |____| |  __/ | | | | |  __/ | | | |_  | (_| | (_| (_|  __/\__ \__ \
     *  |______|_|\___|_| |_| |_|\___|_| |_|\__|  \__,_|\___\___\___||___/___/
     */

    /**
     * Get a single element from the matrix.
     * @param {number[]} index   Zero-based index
     * @return {*} value
     */
    get (index: number[]): Scalar {
        if (!isArray(index)) { throw new TypeError('Array expected') }
        if (index.length !== this.#size.length) { throw new DimensionError(index.length, this.#size.length) }

        // check index
        for (let x = 0; x < index.length; x++) { validateIndex(index[x], this.#size[x]) }

        let data: MultidimArray<Scalar> | Scalar = this.#data
        for (let i = 0, ii = index.length; i < ii; i++) {
            if (!Array.isArray(data)) throw new InternalError('Expected array.');

            const indexI = index[i]
            validateIndex(indexI, data.length)

            data = data[indexI]
        }
        if (Array.isArray(data)) throw new InternalError('Expected scalar.');

        return data
    }
    /**
     * Get a single element from the matrix.
     * @param {DenseMatrix} m    The matrix
     * @param {number[]} index   Zero-based index
     * @return {*} value
     */
    static get<ScalarArithmetics extends Ring<any>>(m: DenseMatrix<ScalarArithmetics>, index: number[]) {
        return m.get(index);
    }


    /**
     * Replace a single element in the matrix.
     * @memberof DenseMatrix
     * @param {number[]} index   Zero-based index
     * @param {*} value          Value to set
     * @return {DenseMatrix} self
     */
    set (index: number[], value: Scalar): this {
        if (!isArray(index)) { throw new TypeError('Array expected') }
        if (index.length < this.#size.length) { throw new DimensionError(index.length, this.#size.length, '<') }

        type Scalar = InstanceOf<ScalarArithmetics>
        const lastI = index[index.length - 1]
        const arr = retrieveTopArrayByIndex<Scalar>(this.#data, index)

        arr[lastI] = value

        return this
    }
    /**
     * Replace a single element in the matrix.
     * @memberof DenseMatrix
     * @param {DenseMatrix} m    The matrix
     * @param {number[]} index   Zero-based index
     * @param {*} value          Value to set
     * @return {DenseMatrix} self
     */
    static set<ScalarArithmetics extends Ring<any>>(
        m: DenseMatrix<ScalarArithmetics>,
        index: number[],
        value: InstanceOf<ScalarArithmetics>
    ) {
        return m.set(index, value)
    }


    /**
     * Create a new matrix with the results of the callback function executed on
     * each entry of the matrix.
     * @memberof DenseMatrix
     * @param {Function} callback   The callback function is invoked with three
     *                              parameters: the value of the element, the index
     *                              of the element, and the Matrix being traversed.
     *
     * @return {DenseMatrix} matrix
     */
    map<
        ResultArithmetics extends Ring<any> = Ring<any>,
        Result extends InstanceOf<ResultArithmetics> = InstanceOf<ResultArithmetics>
    >(
        this: DenseMatrix<ScalarArithmetics, Scalar>,
        callback: (
            value: Scalar,
            index: number[],
            matrix: DenseMatrix<ScalarArithmetics, Scalar>
        ) => Result
    ): DenseMatrix<ResultArithmetics, Result> {

        const self = this
        const recurse = (value: Scalar|MultidimArray<Scalar>, index: number[]): Result|MultidimArray<Result> => {
            if (isArray(value)) {
                return value.map( (child, i) => recurse(child, index.concat(i)) ) as MultidimArray<Result>
            } else {
                return callback(value, index, self)
            }
        }

        const data = recurse(this.#data, []) as MultidimArray<Result>
        return new DenseMatrix(data)
    }
    static map<
        SourceArithmetics extends Ring<any>,
        ResultArithmetics extends Ring<any>,
    >(
        m: DenseMatrix<SourceArithmetics>,
        callback: (
            value: InstanceOf<SourceArithmetics>,
            index: number[],
            matrix: DenseMatrix<SourceArithmetics>
        ) => InstanceOf<ResultArithmetics>
    ): DenseMatrix<ResultArithmetics> {
        return m.map(callback)
    }


    /**
     * Execute a callback function on each entry of the matrix.
     * @memberof DenseMatrix
     * @param {Function} callback   The callback function is invoked with three
     *                              parameters: the value of the element, the index
     *                              of the element, and the Matrix being traversed.
     */
    forEach (callback: (value: Scalar, index: number[], matrix: DenseMatrix<ScalarArithmetics, Scalar>) => void) {

        const self = this
        const recurse = (value: Scalar|MultidimArray<Scalar>, index: number[]) => {
            if (isArray(value)) {
                value.forEach( (child, i) => recurse(child, index.concat(i)) )
            } else {
                callback(value, index, self)
            }
        }
        recurse(this.#data, [])
    }
    static forEach<ScalarArithmetics extends Ring<any>>(
        m: DenseMatrix<ScalarArithmetics>,
        callback: (value: InstanceOf<ScalarArithmetics>, index: number[], matrix: DenseMatrix<ScalarArithmetics>) => void
    ){
        m.forEach(callback)
    }


    /**
     * Iterate over the matrix elements
     */
    [Symbol.iterator] = function* (
        this: DenseMatrix<ScalarArithmetics, Scalar>
    ): IterableIterator<{ value: Scalar, index: number[] }> {
        const recurse = function* (value: Scalar|MultidimArray<Scalar>, index: number[]): any {
            if (isArray(value)) {
                for (let i = 0; i < value.length; i++) {
                    yield* recurse(value[i], index.concat(i))
                }
            } else {
                yield ({ value, index })
            }
        }
        yield* recurse(this.#data, [])
    }


    /**
     * Returns an array containing the rows of a 2D matrix
     * @returns {Array<DenseMatrix>}
     */
    rows (): DenseMatrix<ScalarArithmetics, Scalar>[] {
        const result = []

        const s = this.size
        if (s.length !== 2) {
            throw new TypeError('Rows can only be returned for a 2D matrix.')
        }

        const data = this.#data
        for (const row of data) {
            result.push(new DenseMatrix<ScalarArithmetics, Scalar>([row as Scalar[]]))
        }

        return result
    }
    /**
     * Returns an array containing the rows of a 2D matrix
     * @returns {Array<DenseMatrix>}
     */
    static rows<ScalarArithmetics extends Ring<any>>(m: DenseMatrix<ScalarArithmetics>) {
        return m.rows()
    }


    /**
     * Returns an array containing the columns of a 2D matrix
     * @returns {Array<Matrix>}
     */
    columns (): DenseMatrix<ScalarArithmetics, Scalar>[] {
        const result = []

        const s = this.size
        if (s.length !== 2) {
            throw new TypeError('Rows can only be returned for a 2D matrix.')
        }

        const data = this.#data as Scalar[][]
        for (let i = 0; i < s[1]; i++) {
            const col = data.map(row => [row[i]])
            result.push(new DenseMatrix<ScalarArithmetics, Scalar>(col))
        }

        return result
    }
    /**
     * Returns an array containing the columns of a 2D matrix
     * @returns {Array<Matrix>}
     */
    static columns<ScalarArithmetics extends Ring<any>>(m: DenseMatrix<ScalarArithmetics>) {
        return m.columns()
    }


    /**
     * Create an Array with a copy of the data of the DenseMatrix
     * @memberof DenseMatrix
     * @returns {Array} array
     */
    toArray(): MultidimArray<Scalar> {
        return clone(this.#data)
    }


    /**
     * Get the primitive value of the DenseMatrix: a multidimensional array
     * @memberof DenseMatrix
     * @returns {Array} array
     */
    valueOf(): MultidimArray<Scalar> {
        return this.#data
    }
    // #endregion


    // #region Basic matrix methods
    /*
     *   ____            _                         _        _                        _   _               _
     *  |  _ \          (_)                       | |      (_)                      | | | |             | |
     *  | |_) | __ _ ___ _  ___    _ __ ___   __ _| |_ _ __ ___  __   _ __ ___   ___| |_| |__   ___   __| |___
     *  |  _ < / _` / __| |/ __|  | '_ ` _ \ / _` | __| '__| \ \/ /  | '_ ` _ \ / _ \ __| '_ \ / _ \ / _` / __|
     *  | |_) | (_| \__ \ | (__   | | | | | | (_| | |_| |  | |>  <   | | | | | |  __/ |_| | | | (_) | (_| \__ \
     *  |____/ \__,_|___/_|\___|  |_| |_| |_|\__,_|\__|_|  |_/_/\_\  |_| |_| |_|\___|\__|_| |_|\___/ \__,_|___/
     */

    scale(s: Scalar): DenseMatrix<ScalarArithmetics, Scalar> {
        return this.map(x => this.scalarArithmetics.mul(s, x))
    }
    static scale<ScalarArithmetics extends Ring<any>>(
        m: DenseMatrix<ScalarArithmetics>,
        s: InstanceOf<ScalarArithmetics>
    ){
        return m.scale(s)
    }

    neg(): DenseMatrix<ScalarArithmetics, Scalar> {
        return this.map(x => this.scalarArithmetics.neg(x))
    }
    static neg<ScalarArithmetics extends Ring<any>>(
        m: DenseMatrix<ScalarArithmetics>
    ) {
        return m.neg()
    }


    add(m: DenseMatrix<ScalarArithmetics, Scalar>) {
        return this.map(
            (x, index) => this.scalarArithmetics.add(x, m.get(index))
        )
    }
    static add<ScalarArithmetics extends Ring<any>>(
        a: DenseMatrix<ScalarArithmetics>, b: DenseMatrix<ScalarArithmetics>
    ) {
        return a.add(b)
    }

    sub(m: DenseMatrix<ScalarArithmetics, Scalar>) {
        return this.map(
            (x, index) => this.scalarArithmetics.sub(x, m.get(index))
        )
    }
    static sub<ScalarArithmetics extends Ring<any>>(
        a: DenseMatrix<ScalarArithmetics>, b: DenseMatrix<ScalarArithmetics>
    ) {
        return a.sub(b)
    }


    norm (p?: number | '-inf' | 'inf' | 'fro'): Scalar {
        return norm(this, p)
    }
    static norm<ScalarArithmetics extends Ring<any>>(
        m: DenseMatrix<ScalarArithmetics>,
        p?: number | '-inf' | 'inf' | 'fro'
    ): InstanceOf<ScalarArithmetics> {
        return m.norm(p)
    }

    normSq (p?: number | '-inf' | 'inf' | 'fro'): Scalar {
        return this.scalarArithmetics.pow(this.norm(p), 2)
    }
    static normSq<ScalarArithmetics extends Ring<any>>(
        m: DenseMatrix<ScalarArithmetics>,
        p?: number | '-inf' | 'inf' | 'fro'
    ): InstanceOf<ScalarArithmetics> {
        return m.scalarArithmetics.pow(DenseMatrix.norm<ScalarArithmetics>(m, p), 2)
    }

    /**
     * Right-multiply this matrix by another matrix m
     * @param m
     */
    mul (m: DenseMatrix<ScalarArithmetics, Scalar>) {
        if (this.#size.length !== 2) throw new DimensionError(this.#size.length, 2)
        if (m.#size.length !== 2) throw new DimensionError(m.#size.length, 2)

        const S = this.scalarArithmetics

        const a = this.#data as Scalar[][]
        const b = m.#data as Scalar[][]
        const c: Scalar[][] = []

        const A = a.length
        const B = a[0].length
        const C = b[0].length

        if (b.length !== B) throw new DimensionError(b.length, B)

        for (let i = 0; i < A; i++) {
            c[i] = []

            for (let j = 0; j < C; j++) {
                let r = S.zero()

                for (let k = 0; k < B; k++) {
                    r = S.add(r, S.mul(a[i][k], b[k][j]))
                }

                c[i][j] = r
            }
        }

        return new DenseMatrix<ScalarArithmetics, Scalar>(c)
    }
    // #endregion


    // #region Pointwise operations
    /*
     *   _____      _       _            _                                        _   _
     *  |  __ \    (_)     | |          (_)                                      | | (_)
     *  | |__) |__  _ _ __ | |___      ___ ___  ___     ___  _ __   ___ _ __ __ _| |_ _  ___  _ __  ___
     *  |  ___/ _ \| | '_ \| __\ \ /\ / / / __|/ _ \   / _ \| '_ \ / _ \ '__/ _` | __| |/ _ \| '_ \/ __|
     *  | |  | (_) | | | | | |_ \ V  V /| \__ \  __/  | (_) | |_) |  __/ | | (_| | |_| | (_) | | | \__ \
     *  |_|   \___/|_|_| |_|\__| \_/\_/ |_|___/\___|   \___/| .__/ \___|_|  \__,_|\__|_|\___/|_| |_|___/
     *                                                      | |
     *                                                      |_|
     */

    static dotAdd<ScalarArithmetics extends Ring<any>>(
        a: DenseMatrix<ScalarArithmetics>,
        b: InstanceOf<ScalarArithmetics>
    ): DenseMatrix<ScalarArithmetics> {
        return a.map(el => a.scalarArithmetics.add(el, b))
    }

    static dotSub<ScalarArithmetics extends Ring<any>>(
        a: DenseMatrix<ScalarArithmetics>,
        b: InstanceOf<ScalarArithmetics>
    ): DenseMatrix<ScalarArithmetics> {
        return a.map(el => a.scalarArithmetics.sub(el, b))
    }

    static dotMul<ScalarArithmetics extends Ring<any>>(
        a: DenseMatrix<ScalarArithmetics>,
        b: InstanceOf<ScalarArithmetics>
    ): DenseMatrix<ScalarArithmetics> {
        return a.map(el => a.scalarArithmetics.mul(el, b))
    }

    static dotDiv<ScalarArithmetics extends Ring<any>>(
        a: DenseMatrix<ScalarArithmetics>,
        b: InstanceOf<ScalarArithmetics>
    ): DenseMatrix<ScalarArithmetics> {
        type Scalar = InstanceOf<ScalarArithmetics>
        const A: DivisionRing<Scalar> = a.scalarArithmetics as any
        if (A[symbols.DivisionRing] !== true) throw new ArithmeticError('DivisionRing')
        return a.map(el => A.div(el, b))
    }

    static dotPow<ScalarArithmetics extends Ring<any>>(
        a: DenseMatrix<ScalarArithmetics>,
        b: InstanceOf<ScalarArithmetics>
    ): DenseMatrix<ScalarArithmetics> {
        type Scalar = InstanceOf<ScalarArithmetics>
        const A: NormedDivisionRing<Scalar, any> = a.scalarArithmetics as any
        if (A[symbols.NormedDivisionRing] !== true) throw new ArithmeticError('NormedDivisionRing')
        return a.map(el => A.pow(el, b))
    }

    static dotExp<ScalarArithmetics extends Ring<any>>(a: DenseMatrix<ScalarArithmetics>): DenseMatrix<ScalarArithmetics> {
        return a.map(el => a.scalarArithmetics.exp(el))
    }

    static dotExpm1<ScalarArithmetics extends Ring<any>>(a: DenseMatrix<ScalarArithmetics>): DenseMatrix<ScalarArithmetics> {
        return a.map(el => a.scalarArithmetics.expm1(el))
    }
    // #endregion




    /**
     * Resize the matrix to the given size. Returns a copy of the matrix when
     * `copy=true`, otherwise return the matrix itself (resize in place).
     *
     * @memberof DenseMatrix
     * @param {number[]} size           The new size the matrix should have.
     * @param {*} [defaultValue=0]      Default value, filled in on new entries.
     *                                  If not provided, the matrix elements will
     *                                  be filled with zeros.
     * @param {boolean} [copy]          Return a resized copy of the matrix
     *
     * @return {Matrix}                 The resized matrix
     */
    resize(size: number[], defaultValue: Scalar, copy: boolean = false) {
        const matrix = copy ? this.clone() : this

        matrix.#size = [...size]
        matrix.#data = resize(matrix.#data, matrix.#size, defaultValue)

        return matrix
    }


    /**
     * Get a string representation of the matrix, with optional formatting options.
     * @memberof DenseMatrix
     * @param {Object | number | Function} [options]  Formatting options. See
     *                                                lib/utils/number:format for a
     *                                                description of the available
     *                                                options.
     * @returns {string} str
     */
    format(options: any) {
        return format(this.#data, options)
    }


    /**
     * Get a string representation of the matrix
     * @memberof DenseMatrix
     * @returns {string} str
     */
    toString() {
        return format(this.#data)
    }


    /**
     * Get a JSON representation of the matrix
     * @memberof DenseMatrix
     * @returns {Object}
     */
    toJSON(): SerializedDenseMatrix<ScalarArithmetics> {
        return {
            mathjs: 'DenseMatrix',
            data: this.#data,
            size: this.#size,
            datatype: this.#datatype
        }
    }


    /**
     * Generate a matrix from a JSON object
     * @memberof DenseMatrix
     * @param {Object} json  An object structured like
     *                       `{"mathjs": "DenseMatrix", data: [], size: []}`,
     *                       where mathjs is optional
     * @returns {DenseMatrix}
     */
    static fromJSON(json: any) {
        if (!isSerializedDenseMatrix(json)) throw new TypeError('The object is not a serialized DenseMatrix.')
        return new DenseMatrix(json)
    }


    // TODO
    static kron: any
    static zero: any
    static fromArray: any
    static fromFunction: any
    static equals: any
    static approximatelyEquals: any
    static isFinite: any
    static isNaN: any
    static epsilon: any
}

