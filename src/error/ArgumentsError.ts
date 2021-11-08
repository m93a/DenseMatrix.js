/**
 * Create a syntax error with the message:
 *     'Wrong number of arguments in function <fn> (<count> provided, <min>-<max> expected)'
 * @param {string} fn     Function name
 * @param {number} count  Actual argument count
 * @param {number} min    Minimum required argument count
 * @param {number} [max]  Maximum required argument count
 * @extends Error
 */
export class ArgumentsError extends Error {

     public readonly name = 'ArgumentsError'
     public readonly isArgumentsError = true

     constructor(
          public fn: string,
          public count: number,
          public min: number,
          public max: number
     ) {
          super()

          this.message = `Wrong number of arguments in function ${fn}  (${count} provided, ${
               min + ((max !== undefined && max !== null) ? ('-' + max) : '')} expected)`

          this.stack = (new Error()).stack
     }
}
