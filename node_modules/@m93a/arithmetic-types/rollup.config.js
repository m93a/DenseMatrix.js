import typescript from 'rollup-plugin-typescript2'
import { terser } from 'rollup-plugin-terser'

const name = 'ArithmeticTypes'
const input = 'src/index.ts'
const plugins = [typescript(), terser({ format: { comments: false } })]

const onwarn = ({ message }) => {
    if (/Conflicting re-exports.*"symbols"/.test(message)) return
    if (/Conflicting namespaces.*"symbols"/.test(message)) return
    if (/Circular dependency/.test(message)) return
    console.error(message)
}

export default [
    // ES Modules
    {
        input, plugins, onwarn,
        output: {
            name,
            format: 'es',
            file: 'lib/es/index.js',
        }
    },
    // UMD
    {
        input, plugins, onwarn,
        output: {
            name,
            format: 'umd',
            file: 'lib/umd/index.js',
        }
    },
    // CommonJS
    {
        input, plugins, onwarn,
        output: {
            name,
            format: 'cjs',
            file: 'lib/cjs/index.js',
        }
    },
    // TypeScript Declarations
    {
        input, onwarn,
        plugins: [typescript({
            tsconfigOverride: {
                compilerOptions: {
                    declaration: true,
                    emitDeclarationOnly: true
                }
            }
        })],
        output: {
            name,
            format: 'es',
            file: 'lib/types/index.d.ts'
        },
        onwarn: ({ message }) => {
            if (/Generated an empty chunk: "index"/.test(message)) return
            if (/The emitted file "index.d.ts" overwrites a previously emitted file/.test(message)) return
            console.error(message)
        }
    },
]
