// type checks for all known types
//
// note that:
//
// - check by duck-typing on a property like `isUnit`, instead of checking instanceof.
//   instanceof cannot be used because that would not allow to pass data from
//   one instance of math.js to another since each has it's own instance of Unit.
// - check the `isUnit` property via the constructor, so there will be no
//   matches for "fake" instances like plain objects with a property `isUnit`.
//   That is important for security reasons.
// - It must not be possible to override the type checks used internally,
//   for security reasons, so these functions are not exposed in the expression
//   parser.

export function isNumber (x: any): x is number {
  return typeof x === 'number'
}

export function isBigNumber (x: any) {
  return (x && x.constructor.prototype.isBigNumber === true) || false
}

export function isComplex (x: any) {
  return (x && typeof x === 'object' && Object.getPrototypeOf(x).isComplex === true) || false
}

export function isFraction (x: any) {
  return (x && typeof x === 'object' && Object.getPrototypeOf(x).isFraction === true) || false
}

export function isUnit (x: any) {
  return (x && x.constructor.prototype.isUnit === true) || false
}

export function isString (x: any): x is string {
  return typeof x === 'string'
}

export const isArray = Array.isArray

export function isClonable (x: any): x is { clone(): typeof x } {
  return typeof x?.clone === 'function'
}

export function isDenseMatrix (x: any) {
  return (x && x.isDenseMatrix && x.constructor.prototype.isMatrix === true) || false
}

export function isSparseMatrix (x: any) {
  return (x && x.isSparseMatrix && x.constructor.prototype.isMatrix === true) || false
}

export function isRange (x: any) {
  return (x && x.constructor.prototype.isRange === true) || false
}

export function isIndex (x: any) {
  return (x && x.constructor.prototype.isIndex === true) || false
}

export function isBoolean (x: any) {
  return typeof x === 'boolean'
}

export function isResultSet (x: any) {
  return (x && x.constructor.prototype.isResultSet === true) || false
}

export function isHelp (x: any) {
  return (x && x.constructor.prototype.isHelp === true) || false
}

export function isFunction (x: any): x is (...args: any[]) => any {
  return typeof x === 'function'
}

export function isDate (x: any): x is Date {
  return x instanceof Date
}

export function isRegExp (x: any): x is RegExp {
  return x instanceof RegExp
}

export function isObject (x: any): x is object {
  return !!(x &&
    typeof x === 'object' &&
    x.constructor === Object &&
    !isComplex(x) &&
    !isFraction(x))
}

export function isNull (x: any): x is null {
  return x === null
}

export function isUndefined (x: any): x is undefined {
  return x === undefined
}

export function isAccessorNode (x: any) {
  return (x && x.isAccessorNode === true && x.constructor.prototype.isNode === true) || false
}

export function isArrayNode (x: any) {
  return (x && x.isArrayNode === true && x.constructor.prototype.isNode === true) || false
}

export function isAssignmentNode (x: any) {
  return (x && x.isAssignmentNode === true && x.constructor.prototype.isNode === true) || false
}

export function isBlockNode (x: any) {
  return (x && x.isBlockNode === true && x.constructor.prototype.isNode === true) || false
}

export function isConditionalNode (x: any) {
  return (x && x.isConditionalNode === true && x.constructor.prototype.isNode === true) || false
}

export function isConstantNode (x: any) {
  return (x && x.isConstantNode === true && x.constructor.prototype.isNode === true) || false
}

export function isFunctionAssignmentNode (x: any) {
  return (x && x.isFunctionAssignmentNode === true && x.constructor.prototype.isNode === true) || false
}

export function isFunctionNode (x: any) {
  return (x && x.isFunctionNode === true && x.constructor.prototype.isNode === true) || false
}

export function isIndexNode (x: any) {
  return (x && x.isIndexNode === true && x.constructor.prototype.isNode === true) || false
}

export function isNode (x: any) {
  return (x && x.isNode === true && x.constructor.prototype.isNode === true) || false
}

export function isObjectNode (x: any) {
  return (x && x.isObjectNode === true && x.constructor.prototype.isNode === true) || false
}

export function isOperatorNode (x: any) {
  return (x && x.isOperatorNode === true && x.constructor.prototype.isNode === true) || false
}

export function isParenthesisNode (x: any) {
  return (x && x.isParenthesisNode === true && x.constructor.prototype.isNode === true) || false
}

export function isRangeNode (x: any) {
  return (x && x.isRangeNode === true && x.constructor.prototype.isNode === true) || false
}

export function isSymbolNode (x: any) {
  return (x && x.isSymbolNode === true && x.constructor.prototype.isNode === true) || false
}

export function isChain (x: any) {
  return (x && x.constructor.prototype.isChain === true) || false
}

export function typeOf (x: any) {
  const t = typeof x

  if (t === 'object') {
    // JavaScript types
    if (x === null) return 'null'
    if (Array.isArray(x)) return 'Array'
    if (x instanceof Date) return 'Date'
    if (x instanceof RegExp) return 'RegExp'

    // math.js types
    if (isBigNumber(x)) return 'BigNumber'
    if (isComplex(x)) return 'Complex'
    if (isFraction(x)) return 'Fraction'
    if (isUnit(x)) return 'Unit'
    if (isIndex(x)) return 'Index'
    if (isRange(x)) return 'Range'
    if (isResultSet(x)) return 'ResultSet'
    if (isNode(x)) return x.type
    if (isChain(x)) return 'Chain'
    if (isHelp(x)) return 'Help'

    return 'Object'
  }

  if (t === 'function') return 'Function'

  return t // can be 'string', 'number', 'boolean', ...
}
