// GitHub Pages serves 404.html for unknown paths. Making it a copy of the app
// shell lets deep links such as /tracklens/model load and route client-side.
import { copyFileSync } from 'node:fs'
import { resolve } from 'node:path'

const out = resolve(process.argv[2] ?? 'dist')
copyFileSync(resolve(out, 'index.html'), resolve(out, '404.html'))
console.log(`postbuild: wrote ${out}/404.html`)
