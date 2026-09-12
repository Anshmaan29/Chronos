# assets

## fonts/

Every face the console draws with is bundled here and registered with Qt at
startup by `chronos.ui.theme.load_fonts()`. Nothing is requested from the
system and nothing is downloaded at runtime.

This is deliberate. `Impact` and `Arial Black` are not installed on every
machine, and the previous stylesheets named them first. A font fallback
discovered live, on a projector, on a machine that is not yours, is an
avoidable way to lose a demo — and a silently substituted face changes every
measured text width in the layout, so things clip rather than merely look
different.

| file | family | used for | licence |
|---|---|---|---|
| `IBMPlexMono-Regular.ttf`, `-SemiBold.ttf` | IBM Plex Mono | **every number, in every theme** | OFL 1.1 |
| `TitilliumWeb-Regular/-SemiBold/-Bold/-Black.ttf` | Titillium Web | `f1` theme labels and prose | OFL 1.1 |
| `ArchivoBlack-Regular.ttf` | Archivo Black | the wordmark, `brutal` headings | OFL 1.1 |
| `Anton-Regular.ttf` | Anton | condensed fallback | OFL 1.1 |

Titillium is the face the Formula 1 wordmark and timing graphics are drawn
from, which is why it carries the `f1` theme: it is the closest openly
licensed match to the broadcast look.

Full licence text sits beside each family as `OFL-<Family>.txt`. The SIL Open
Font License permits bundling and redistribution, including commercially,
provided the licence travels with the files — which is what those four files
are for. Do not delete them.

The CSS stacks in `theme.py` still name system fallbacks *after* the bundled
family, so a missing file degrades to something readable instead of crashing.
`load_fonts()` returns a one-line report naming anything it could not load,
and the console prints it at startup, so a missing face is visible in the
first second rather than at the worst moment.

## logo.png — absent on purpose

`chronos.ui.widgets.Wordmark` looks for `assets/logo.png` and draws it instead
of the built-in mark if it is there.

Nothing is shipped at that path. The Formula 1 roundel is a registered trade
mark of Formula One Licensing BV, and bundling it inside an MIT-licensed
public repository is a liability the project does not need — particularly in
front of judges who work in that world and will recognise it as borrowed.

So the console draws its own mark in the same visual grammar: angled speed
slashes in brand red, heavy italic type, the carbon-black ground. Same
language, nothing taken.

If you decide you want a different mark, drop a PNG at `assets/logo.png` and
it is picked up on the next launch. That is a decision worth making knowingly
rather than inheriting from a default.
