/**
 * Create a range error with the message:
 *     'Dimension mismatch (<actual size> != <expected size>)'
 * @param {number | number[]} actual        The actual size
 * @param {number | number[]} expected      The expected size
 * @param {string} [relation='!=']          Optional relation between actual
 *                                          and expected size: '!=', '<', etc.
 * @extends RangeError
 */
export class DimensionError extends RangeError {
  public readonly name = 'DimensionError'
  public readonly isDimensionError = true

  constructor (
    public actual: number | number[],
    public expected: number | number[],
    public relation: string = '!='
  ) {
    super()

    this.message = 'Dimension mismatch (' +
        (Array.isArray(actual) ? ('[' + actual.join(', ') + ']') : actual) +
        ' ' + relation + ' ' +
        (Array.isArray(expected) ? ('[' + expected.join(', ') + ']') : expected) +
        ')'

  }
}
