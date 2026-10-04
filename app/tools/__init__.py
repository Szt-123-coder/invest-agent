from .alerts import delete_alert, list_alerts, remember_preference, set_move_alert, set_price_alert, watch
from .market import compare_with_index, find_similar_history, get_history, get_quote
from .news import search_news

ALL_TOOLS = [get_quote, get_history, find_similar_history, compare_with_index, search_news, set_price_alert, set_move_alert, list_alerts,
             delete_alert, watch, remember_preference]
