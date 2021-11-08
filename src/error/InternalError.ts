/**
 * Create a syntax error with the message:
 *     '<message>. This is not a mistake on your part, something went wrong in a 3rd party code.'
 * @param {string} message     Function name
 * @extends Error
 */
export class InternalError extends Error {

    public readonly name = 'ArgumentsError'
    public readonly isArgumentsError = true

    constructor(message: string) {
        super()

        this.message = `${message}. This is not a mistake on your part, something went wrong in a 3rd party code.`

        this.stack = (new Error()).stack
    }
}
