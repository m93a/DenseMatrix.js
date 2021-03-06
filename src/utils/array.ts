import { isInteger } from './number'
import { isNumber } from './is'
import { format } from './string'
import { DimensionError } from '../error/DimensionError'
import { IndexError } from '../error/IndexError'

export type MultidimArray<R> = R[] | MultidimArray<R>[]

/**
 * Calculate the size of a multi dimensional array.
 * This function checks the size of the first entry, it does not validate
 * whether all dimensions match. (use function `validate` for that)
 * @param {Array} x
 * @Return {Number[]} size
 */
export function arraySize (x: MultidimArray<any>) {
  const s = []

  while (Array.isArray(x)) {
    s.push(x.length)
    x = x[0]
  }

  return s
}

/**
 * Recursively validate whether each element in a multi dimensional array
 * has a size corresponding to the provided size array.
 * @param {Array} array    Array to be validated
 * @param {number[]} size  Array with the size of each dimension
 * @param {number} dim   Current dimension
 * @throws DimensionError
 * @private
 */
function _validate (array: any[], size: number[], dim: number) {
  let i
  const len = array.length

  if (len !== size[dim]) {
    throw new DimensionError(len, size[dim])
  }

  if (dim < size.length - 1) {
    // recursively validate each child array
    const dimNext = dim + 1
    for (i = 0; i < len; i++) {
      const child = array[i]
      if (!Array.isArray(child)) {
        throw new DimensionError(size.length - 1, size.length, '<')
      }
      _validate(array[i], size, dimNext)
    }
  } else {
    // last dimension. none of the childs may be an array
    for (i = 0; i < len; i++) {
      if (Array.isArray(array[i])) {
        throw new DimensionError(size.length + 1, size.length, '>')
      }
    }
  }
}

/**
 * Validate whether each element in a multi dimensional array has
 * a size corresponding to the provided size array.
 * @param {Array} array    Array to be validated
 * @param {number[]} size  Array with the size of each dimension
 * @throws DimensionError
 */
export function validate (array: any[], size: number[]) {
  const isScalar = (size.length === 0)
  if (isScalar) {
    // scalar
    if (Array.isArray(array)) {
      throw new DimensionError(array.length, 0)
    }
  } else {
    // array
    _validate(array, size, 0)
  }
}

/**
 * Test whether index is an integer number with index >= 0 and index < length
 * when length is provided
 * @param {number} index    Zero-based index
 * @param {number} [length] Length of the array
 */
export function validateIndex (index: number, length: number) {
  if (!isNumber(index) || !isInteger(index)) {
    throw new TypeError('Index must be an integer (value: ' + index + ')')
  }
  if (index < 0 || (typeof length === 'number' && index >= length)) {
    throw new IndexError(index, length)
  }
}

/**
 * Resize a multi dimensional array. The resized array is returned.
 * @param {Array} array         Array to be resized
 * @param {Array.<number>} size Array with the size of each dimension
 * @param {*} [defaultValue=0]  Value to be filled in in new entries,
 *                              zero by default. Specify for example `null`,
 *                              to clearly see entries that are not explicitly
 *                              set.
 * @return {Array} array         The resized array
 */
export function resize (array: any[], size: number, defaultValue: any) {
  // TODO: add support for scalars, having size=[] ?

  // check the type of the arguments
  if (!Array.isArray(array) || !Array.isArray(size)) {
    throw new TypeError('Array expected')
  }
  if (size.length === 0) {
    throw new Error('Resizing to scalar is not supported')
  }

  // check whether size contains positive integers
  size.forEach(function (value) {
    if (!isNumber(value) || !isInteger(value) || value < 0) {
      throw new TypeError('Invalid size, must contain positive integers ' +
        '(size: ' + format(size) + ')')
    }
  })

  // recursively resize the array
  const _defaultValue = (defaultValue !== undefined) ? defaultValue : 0
  _resize(array, size, 0, _defaultValue)

  return array
}

/**
 * Recursively resize a multi dimensional array
 * @param {Array} array         Array to be resized
 * @param {number[]} size       Array with the size of each dimension
 * @param {number} dim          Current dimension
 * @param {*} [defaultValue]    Value to be filled in in new entries,
 *                              undefined by default.
 * @private
 */
function _resize (array: any[], size: number[], dim: number, defaultValue: any) {
  const oldLen = array.length
  const newLen = size[dim]
  const minLen = Math.min(oldLen, newLen)

  // apply new length
  array.length = newLen

  if (dim < size.length - 1) {
    // non-last dimension
    const dimNext = dim + 1

    // resize existing child arrays
    for (let i = 0; i < minLen; i++) {
      // resize child array
      let elem = array[i]
      if (!Array.isArray(elem)) {
        elem = [elem] // add a dimension
        array[i] = elem
      }
      _resize(elem, size, dimNext, defaultValue)
    }

    // create new child arrays
    for (let i = minLen; i < newLen; i++) {
      // get child array
      let elem: any = []
      array[i] = elem

      // resize new child array
      _resize(elem, size, dimNext, defaultValue)
    }
  } else {
    // last dimension

    // remove dimensions of existing values
    for (let i = 0; i < minLen; i++) {
      while (Array.isArray(array[i])) {
        array[i] = array[i][0]
      }
    }

    // fill new elements with the default value
    for (let i = minLen; i < newLen; i++) {
      array[i] = defaultValue
    }
  }
}

/**
 * Re-shape a multi dimensional array to fit the specified dimensions
 * @param {Array} array           Array to be reshaped
 * @param {Array.<number>} sizes  List of sizes for each dimension
 * @returns {Array}               Array whose data has been formatted to fit the
 *                                specified dimensions
 *
 * @throws {DimensionError}       If the product of the new dimension sizes does
 *                                not equal that of the old ones
 */
export function reshape (array: any[], sizes: number[]) {
  const flatArray = flatten(array)
  const currentLength = flatArray.length

  if (!Array.isArray(array) || !Array.isArray(sizes)) {
    throw new TypeError('Array expected')
  }

  if (sizes.length === 0) {
    throw new DimensionError(0, currentLength, '!=')
  }

  sizes = processSizesWildcard(sizes, currentLength)
  const newLength = product(sizes)
  if (currentLength !== newLength) {
    throw new DimensionError(
      newLength,
      currentLength,
      '!='
    )
  }

  try {
    return _reshape(flatArray, sizes)
  } catch (e) {
    if (e instanceof DimensionError) {
      throw new DimensionError(
        newLength,
        currentLength,
        '!='
      )
    }
    throw e
  }
}

/**
 * Replaces the wildcard -1 in the sizes array.
 * @param {Array.<number>} sizes  List of sizes for each dimension. At most on wildcard.
 * @param {number} currentLength  Number of elements in the array.
 * @throws {Error}                If more than one wildcard or unable to replace it.
 * @returns {Array.<number>}      The sizes array with wildcard replaced.
 */
export function processSizesWildcard (sizes: number[], currentLength: number) {
  const newLength = product(sizes)
  const processedSizes = sizes.slice()
  const WILDCARD = -1
  const wildCardIndex = sizes.indexOf(WILDCARD)

  const isMoreThanOneWildcard = sizes.indexOf(WILDCARD, wildCardIndex + 1) >= 0
  if (isMoreThanOneWildcard) {
    throw new Error('More than one wildcard in sizes')
  }

  const hasWildcard = wildCardIndex >= 0
  const canReplaceWildcard = currentLength % newLength === 0

  if (hasWildcard) {
    if (canReplaceWildcard) {
      processedSizes[wildCardIndex] = -currentLength / newLength
    } else {
      throw new Error('Could not replace wildcard, since ' + currentLength + ' is no multiple of ' + (-newLength))
    }
  }
  return processedSizes
}

/**
 * Computes the product of all array elements.
 * @param {Array<number>} array Array of factors
 * @returns {number}            Product of all elements
 */
function product (array: number[]) {
  return array.reduce((prev, curr) => prev * curr, 1)
}

/**
 * Iteratively re-shape a multi dimensional array to fit the specified dimensions
 * @param {Array} array           Array to be reshaped
 * @param {Array.<number>} sizes  List of sizes for each dimension
 * @returns {Array}               Array whose data has been formatted to fit the
 *                                specified dimensions
 */

function _reshape (array: any[], sizes: number[]) {
  // testing if there are enough elements for the requested shape
  let tmpArray = array
  let tmpArray2
  // for each dimensions starting by the last one and ignoring the first one
  for (let sizeIndex = sizes.length - 1; sizeIndex > 0; sizeIndex--) {
    const size = sizes[sizeIndex]
    tmpArray2 = []

    // aggregate the elements of the current tmpArray in elements of the requested size
    const length = tmpArray.length / size
    for (let i = 0; i < length; i++) {
      tmpArray2.push(tmpArray.slice(i * size, (i + 1) * size))
    }
    // set it as the new tmpArray for the next loop turn or for return
    tmpArray = tmpArray2
  }

  return tmpArray
}

/**
 * Squeeze a multi dimensional array
 * @param {Array} array
 * @param {Array} [size]
 * @returns {Array} returns the array itself
 */
export function squeeze (array: any[], size: any[]) {
  const s = size || arraySize(array)

  // squeeze outer dimensions
  while (Array.isArray(array) && array.length === 1) {
    array = array[0]
    s.shift()
  }

  // find the first dimension to be squeezed
  let dims = s.length
  while (s[dims - 1] === 1) {
    dims--
  }

  // squeeze inner dimensions
  if (dims < s.length) {
    array = _squeeze(array, dims, 0)
    s.length = dims
  }

  return array
}

/**
 * Recursively squeeze a multi dimensional array
 * @param {Array} array
 * @param {number} dims Required number of dimensions
 * @param {number} dim  Current dimension
 * @returns {Array | *} Returns the squeezed array
 * @private
 */
function _squeeze (array: any[], dims: number, dim: number) {
  let i, ii

  if (dim < dims) {
    const next = dim + 1
    for (i = 0, ii = array.length; i < ii; i++) {
      array[i] = _squeeze(array[i], dims, next)
    }
  } else {
    while (Array.isArray(array)) {
      array = array[0]
    }
  }

  return array
}

/**
 * Unsqueeze a multi dimensional array: add dimensions when missing
 *
 * Paramter `size` will be mutated to match the new, unqueezed matrix size.
 *
 * @param {Array} array
 * @param {number} dims       Desired number of dimensions of the array
 * @param {number} [outer]    Number of outer dimensions to be added
 * @param {Array} [size] Current size of array.
 * @returns {Array} returns the array itself
 * @private
 */
export function unsqueeze (array: any[], dims: number, outer?: number, size?: any[]) {
  const s = size || arraySize(array)

  // unsqueeze outer dimensions
  if (outer) {
    for (let i = 0; i < outer; i++) {
      array = [array]
      s.unshift(1)
    }
  }

  // unsqueeze inner dimensions
  array = _unsqueeze(array, dims, 0)
  while (s.length < dims) {
    s.push(1)
  }

  return array
}

/**
 * Recursively unsqueeze a multi dimensional array
 * @param {Array} array
 * @param {number} dims Required number of dimensions
 * @param {number} dim  Current dimension
 * @returns {Array | *} Returns the squeezed array
 * @private
 */
function _unsqueeze (array: any[], dims: number, dim: number) {
  let i, ii

  if (Array.isArray(array)) {
    const next = dim + 1
    for (i = 0, ii = array.length; i < ii; i++) {
      array[i] = _unsqueeze(array[i], dims, next)
    }
  } else {
    for (let d = dim; d < dims; d++) {
      array = [array]
    }
  }

  return array
}
/**
 * Flatten a multi dimensional array, put all elements in a one dimensional
 * array
 * @param {Array} array   A multi dimensional array
 * @return {Array}        The flattened array (1 dimensional)
 */
export function flatten (array: any[]) {
  if (!Array.isArray(array)) {
    // if not an array, return as is
    return array
  }
  const flat: any[] = []

  array.forEach(function callback (value) {
    if (Array.isArray(value)) {
      value.forEach(callback) // traverse through sub-arrays recursively
    } else {
      flat.push(value)
    }
  })

  return flat
}

/**
 * Filter values in a callback given a regular expression
 * @param {Array} array
 * @param {RegExp} regexp
 * @return {Array} Returns the filtered array
 * @private
 */
export function filterRegExp (array: any[], regexp: RegExp) {
  if (arraySize(array).length !== 1) {
    throw new Error('Only one dimensional matrices supported')
  }

  return Array.prototype.filter.call(array, (entry) => regexp.test(entry))
}

/**
 * Assign a numeric identifier to every element of a sorted array
 * @param {Array} a  An array
 * @return {Array} An array of objects containing the original value and its identifier
 */
export function identify (a: any[]) {
  if (!Array.isArray(a)) {
    throw new TypeError('Array input expected')
  }

  if (a.length === 0) {
    return a
  }

  const b = []
  let count = 0
  b[0] = { value: a[0], identifier: 0 }
  for (let i = 1; i < a.length; i++) {
    if (a[i] === a[i - 1]) {
      count++
    } else {
      count = 0
    }
    b.push({ value: a[i], identifier: count })
  }
  return b
}

/**
 * Remove the numeric identifier from the elements
 * @param {array} a  An array
 * @return {array} An array of values without identifiers
 */
export function generalize (a: any[]) {
  if (!Array.isArray(a)) {
    throw new TypeError('Array input expected')
  }

  if (a.length === 0) {
    return a
  }

  const b = []
  for (let i = 0; i < a.length; i++) {
    b.push(a[i].value)
  }
  return b
}

/**
 * Check the datatype of a given object
 * This is a low level implementation that should only be used by
 * parent Matrix classes such as SparseMatrix or DenseMatrix
 * This method does not validate Array Matrix shape
 * @param {Array} array
 * @param {function} typeOf   Callback function to use to determine the type of a value
 * @return {string}
 */
export function getArrayDataType (array: any, typeOf: (x: any) => string): string | undefined {
  let type // to hold type info
  let length = 0 // to hold length value to ensure it has consistent sizes

  for (let i = 0; i < array.length; i++) {
    const item = array[i]
    const isArray = Array.isArray(item)

    // Saving the target matrix row size
    if (i === 0 && isArray) {
      length = item.length
    }

    // If the current item is an array but the length does not equal the targetVectorSize
    if (isArray && item.length !== length) {
      return undefined
    }

    const itemType = isArray
      ? getArrayDataType(item, typeOf) // recurse into a nested array
      : typeOf(item)

    if (type === undefined) {
      type = itemType // first item
    } else if (type !== itemType) {
      return 'mixed'
    } else {
      // we're good, everything has the same type so far
    }
  }

  return type
}

/**
 * Return the last item from an array
 * @param array
 * @returns {*}
 */
export function last (array: any[]) {
  return array[array.length - 1]
}

/**
 * Get all but the last element of array.
 */
export function initial (array: any[]) {
  return array.slice(0, array.length - 1)
}

/**
 * Test whether an array or string contains an item
 * @param {Array | string} array
 * @param {*} item
 * @return {boolean}
 */
export function contains (array: any[], item: any) {
  return array.indexOf(item) !== -1
}
