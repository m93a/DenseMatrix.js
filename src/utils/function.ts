// function utils

import { lruQueue } from './lruQueue'

interface MemoizeOptions {
  hasher?: (str: object) => string
  limit?: number
}

interface MemoizedFn {
  (...args: any[]): any
  cache: {
    values: Map<any, any>,
    lru: ReturnType<typeof lruQueue>
  }
}

/**
 * Memoize a given function by caching the computed result.
 * The cache of a memoized function can be cleared by deleting the `cache`
 * property of the function.
 *
 * @param {function} fn                     The function to be memoized.
 *                                          Must be a pure function.
 * @param {Object} [options]
 * @param {function(args: Array): string} [options.hasher]
 *    A custom hash builder. Is JSON.stringify by default.
 * @param {number | undefined} [options.limit]
 *    Maximum number of values that may be cached. Undefined indicates
 *    unlimited (default)
 * @return {function}                       Returns the memoized function
 */
export function memoize (fn, options: MemoizeOptions = {}) {
  let { hasher, limit } = options

  if (limit == null) limit = Number.POSITIVE_INFINITY
  if (hasher == null) hasher = JSON.stringify

  const memoized = <MemoizedFn> function (...args: any[]) {
    if (typeof memoized.cache !== 'object') {
      memoized.cache = {
        values: new Map(),
        lru: lruQueue(limit || Number.POSITIVE_INFINITY)
      }
    }
    const hash = hasher(args)

    if (memoized.cache.values.has(hash)) {
      memoized.cache.lru.hit(hash)
      return memoized.cache.values.get(hash)
    }

    const newVal = fn.apply(fn, args)
    memoized.cache.values.set(hash, newVal)
    memoized.cache.values.delete(memoized.cache.lru.hit(hash))

    return newVal
  }

  return memoized
}

/**
 * Memoize a given function by caching all results and the arguments,
 * and comparing against the arguments of previous results before
 * executing again.
 * This is less performant than `memoize` which calculates a hash,
 * which is very fast to compare. Use `memoizeCompare` only when it is
 * not possible to create a unique serializable hash from the function
 * arguments.
 * The isEqual function must compare two sets of arguments
 * and return true when equal (can be a deep equality check for example).
 * @param {function} fn
 * @param {function(a: *, b: *) : boolean} isEqual
 * @returns {function}
 */
export function memoizeCompare (fn, isEqual) {
  const memoize = function memoize () {
    const args = []
    for (let i = 0; i < arguments.length; i++) {
      args[i] = arguments[i]
    }

    for (let c = 0; c < memoize.cache.length; c++) {
      const cached = memoize.cache[c]

      if (isEqual(args, cached.args)) {
        // TODO: move this cache entry to the top so recently used entries move up?
        return cached.res
      }
    }

    const res = fn.apply(fn, args)
    memoize.cache.unshift({ args, res })

    return res
  }

  memoize.cache = []

  return memoize
}

/**
 * Find the maximum number of arguments expected by a typed function.
 * @param {function} fn   A typed function
 * @return {number} Returns the maximum number of expected arguments.
 *                  Returns -1 when no signatures where found on the function.
 */
export function maxArgumentCount (fn) {
  return Object.keys(fn.signatures || {})
    .reduce(function (args, signature) {
      const count = (signature.match(/,/g) || []).length + 1
      return Math.max(args, count)
    }, -1)
}
