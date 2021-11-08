/**
 * Create a range error with the message:
 *     'Index out of range (index < min)'
 *     'Index out of range (index < max)'
 *
 * @param {number} index     The actual index
 * @param {number} [min=0]   Minimum index (included)
 * @param {number} [max]     Maximum index (excluded)
 * @extends RangeError
 */
export class IndexError extends RangeError {
  public readonly name = 'IndexError'
  public readonly isIndexError = true

  public index: number
  public min: number
  public max: number

  constructor (index: number, min?: number, max?: number) {
    super()

    this.index = index
    if (arguments.length < 3) {
      this.min = 0
      this.max = min
    } else {
      this.min = min
      this.max = max
    }

    if (this.min !== undefined && this.index < this.min) {
      this.message = 'Index out of range (' + this.index + ' < ' + this.min + ')'
    } else if (this.max !== undefined && this.index >= this.max) {
      this.message = 'Index out of range (' + this.index + ' > ' + (this.max - 1) + ')'
    } else {
      this.message = 'Index out of range (' + this.index + ')'
    }
  }
}

