from functools import partial
from shiny import module, reactive, render, ui

ui_card_header = partial(ui.card_header, class_="bg-dark")

ui_katex = ui.head_content(
    ui.tags.link(
        rel="stylesheet",
        href="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.css"
    ),
    ui.tags.script(src="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.js"),
    ui.tags.script(src="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/contrib/auto-render.min.js"),
    ui.tags.script("""
    document.addEventListener('DOMContentLoaded', function() {
        renderMathInElement(document.body, {
                delimiters: [
                {left: "$$", right: "$$", display: true},
                {left: "\\[", right: "\\]", display: true},
                {left: "$", right: "$", display: false},
            ]
        });
    });"""
    ),
)

ui_css = ui.head_content(
    ui.tags.link(
        rel="stylesheet",
        href="/css/custom.css"
    )
)
