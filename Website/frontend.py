
from reactpy import component, html, hooks, event, utils, run
from reactpy.backend.starlette import configure, Starlette
import pprint as pp
import backend

BOOTSTRAP_CSS = html.link(
    {
        'rel': 'stylesheet',
        'href': 'https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/'
                'dist/css/bootstrap.min.css',
        'integrity': 'sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Y'
                     'z1ztcQTwFspd3yD65VohhpuuCOmLASjC',
        'crossorigin': 'anonymous',
    }
)

@component
def FormStockTicker(t, set_t, id):
    return html.div(
        {'class': 'mb-3'},
        html.label(
            {
                'class': 'form-label fw-bold',
                'for': 'ticker'
            },
            f"Ticker {id}",
        ),
        html.input(
            {
                'class': 'form-control',
                'type': 'text',
                'id': 'ticker',
                'placeholder': 'enter ticker',
                'on_change': lambda event: set_t(event['target']['value']),
                'value': t,
            },
        ),
    )

@component
def FormSubmitButton():
    return html.input(
        {
            'class': 'btn btn-primary mb-3',
            'type': 'submit',
            'value': 'Submit'
        },
    )

@component
def Index():
    t1, set_t1 = hooks.use_state('')
    t2, set_t2 = hooks.use_state('')
    t3, set_t3 = hooks.use_state('')

    @event(prevent_default=True)
    def submit_tickers(event):
        print(f"Submitted Tickers {t1}, {t2}, {t3}")
        backend.submitTickers(t1, t2, t3)

    return html.div(
        BOOTSTRAP_CSS,
        html.div(
            {'class': 'container mt-3'},
            html.div(
                {'class': 'row'},
                html.div(
                    {'class': 'col-lg-6'},
                    html.form(
                        {'on_submit': submit_tickers},
                        FormStockTicker(t1, set_t1, 1),
                        FormStockTicker(t2, set_t2, 2),
                        FormStockTicker(t3, set_t3, 3),
                        FormSubmitButton()
                    ),
                ),
            ),
        ),
    )

run(Index, port=7000)
