from gym_trading_env.renderer import Renderer   
from BitcoinMetrics import add_render_metrics


def render_saved_logs(render_dir: str):
    renderer = Renderer(render_logs_dir=render_dir)
    add_render_metrics(renderer)
    renderer.run()

if __name__ == "__main__":
    render_saved_logs('render_logs')
