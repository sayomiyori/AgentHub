import json
from unittest.mock import Mock

import pytest
import redis

from app.cache import semantic_cache as cache


@pytest.mark.parametrize('operation', ['get', 'set'])
def test_redis_loss_is_optional_cache_miss(monkeypatch, operation):
    client = Mock()
    client.get.side_effect = redis.ConnectionError('synthetic outage')
    monkeypatch.setattr(cache, '_client', client)
    params = dict(top_k=5, provider='test', model='test')
    if operation == 'get':
        assert cache.get_cached_rag_answer('question', [1.0], **params) is None
    else:
        cache.set_cached_rag_answer('question', [1.0], payload={'answer': 'answer'}, **params)


@pytest.mark.parametrize('raw', ['null', '{}', '[null]', '[{"embedding":["bad"]}]', '[{"embedding":[NaN]}]'])
def test_malformed_cache_entries_are_misses(monkeypatch, raw):
    client = Mock()
    client.get.return_value = raw
    monkeypatch.setattr(cache, '_client', client)
    assert cache.get_cached_rag_answer('question', [1.0], top_k=5, provider='test', model='test') is None


def test_embeddings_of_different_dimensions_do_not_match():
    assert cache._cosine([1.0], [1.0, 100.0]) == 0.0


def test_cache_hit_does_not_renew_age(monkeypatch):
    import time
    client = Mock()
    client.get.return_value = json.dumps([dict(embedding=[1.0], meta=dict(top_k=5, provider='p', model='m'),
                                              payload={'answer': 'answer', 'sources': []}, ts=time.time())])
    monkeypatch.setattr(cache, '_client', client)
    assert cache.get_cached_rag_answer('q', [1.0], top_k=5, provider='p', model='m') == {
        'answer': 'answer', 'sources': []}
    client.expire.assert_not_called()


@pytest.mark.parametrize('change', [
    'expired', 'nan', 'dimension', 'meta', 'payload', 'huge', 'timestamp', 'empty', 'source', 'surrogate', 'nul',
])
def test_corruption_with_valid_timestamp_is_ignored(monkeypatch, change):
    import time
    entry = dict(embedding=[1.0], meta=dict(top_k=5, provider='p', model='m'),
                 payload={'answer': 'answer', 'sources': []}, ts=time.time())
    if change == 'expired':
        entry['ts'] -= cache.TTL_SECONDS + 1
    elif change == 'nan':
        entry['embedding'] = [float('nan')]
    elif change == 'dimension':
        entry['embedding'] = [1.0, 1.0]
    elif change == 'meta':
        entry['meta'] = 5
    elif change == 'huge':
        entry['embedding'] = [10 ** 400]
    elif change == 'timestamp':
        entry['ts'] = 10 ** 400
    elif change == 'empty':
        entry['payload'] = {}
    elif change == 'source':
        entry['payload']['sources'] = [{}]
    elif change == 'surrogate':
        entry['payload']['answer'] = '\ud800'
    elif change == 'nul':
        entry['payload']['answer'] = '\x00'
    else:
        entry['payload'] = None
    client = Mock()
    client.get.return_value = json.dumps([entry])
    monkeypatch.setattr(cache, '_client', client)
    assert cache.get_cached_rag_answer('q', [1.0], top_k=5, provider='p', model='m') is None


def test_cache_write_outage_does_not_discard_answer(monkeypatch):
    client = Mock()
    client.get.return_value = None
    client.set.side_effect = redis.TimeoutError('synthetic timeout')
    monkeypatch.setattr(cache, '_client', client)
    cache.set_cached_rag_answer('q', [1.0], top_k=5, provider='p', model='m', payload={'answer': 'answer'})


def test_metadata_mismatch_skips_vector_math(monkeypatch):
    import time
    client = Mock()
    client.get.return_value = json.dumps([dict(embedding=[1.0], meta=dict(top_k=5, provider='other', model='m'),
                                              payload={'answer': 'answer', 'sources': []}, ts=time.time())])
    cosine = Mock(side_effect=AssertionError('Unnecessary vector comparison'))
    monkeypatch.setattr(cache, '_client', client)
    monkeypatch.setattr(cache, '_cosine', cosine)
    assert cache.get_cached_rag_answer('q', [1.0], top_k=5, provider='p', model='m') is None
    cosine.assert_not_called()
