import { InternalError } from "./InternalError"

/**
 * Create a syntax error with the message:
 *     '<what> is not implemented yet. This is not a mistake on your part, something went wrong in a 3rd party code.'
 * @param {string} what     Function name
 * @extends InternalError
 */
export class NotImplementedError extends InternalError {

    public readonly name = 'ArgumentsError'
    public readonly isArgumentsError = true

    constructor(what: string) {
        super(`${what} is not implemented yet`)
    }
}
