/**
 * Create a type error with the message:
 *     'Unsupported type of arithmetic, expected <expectedArithmetic>'
 * @param {number | number[]} expected      The expected size
 * @param {string} [relation='!=']          Optional relation between actual
 *                                          and expected size: '!=', '<', etc.
 * @extends TypeError
 */
export class ArithmeticError extends TypeError {
    public readonly name = 'ArithmeticError'
    public readonly isArithmeticError = true

    constructor (
      public expectedArithmetic: string
    ) {
      super()

      this.message = `Unsupported type of arithmetic, expected ${expectedArithmetic}`

    }
  }
