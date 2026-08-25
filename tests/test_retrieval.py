from vga.reason import retrieval

# Test tokens function
assert retrieval.tokens("") == set()
assert retrieval.tokens("a an the") == set()
assert retrieval.tokens("hello world") == {"hello", "world"}
assert retrieval.tokens("A quick brown fox jumps over the lazy dog.") == {"quick", "brown", "fox", "jumps", "over", "lazy", "dog"}

# Test rank_by_overlap function
items = [
    {"text": "hello world", "confidence": 0.9, "uses": 10},
    {"text": "foo bar", "confidence": 0.8, "uses": 5},
    {"text": "hello foo", "confidence": 0.7, "uses": 3}
]

context = "hello"
searchable = lambda item: item["text"]
tiebreak = lambda item: (item["confidence"], item["uses"])
limit = 2

assert retrieval.rank_by_overlap(items, context, searchable, tiebreak, limit) == [
    {"text": "hello world", "confidence": 0.9, "uses": 10},
    {"text": "hello foo", "confidence": 0.7, "uses": 3}
]

context = ""
assert retrieval.rank_by_overlap(items, context, searchable, tiebreak, limit) == [
    {"text": "hello world", "confidence": 0.9, "uses": 10},
    {"text": "foo bar", "confidence": 0.8, "uses": 5}
]

context = "unknown"
assert retrieval.rank_by_overlap(items, context, searchable, tiebreak, limit) == []

items = [
    {"text": "hello world", "confidence": 0.9, "uses": 10},
    {"text": "foo bar", "confidence": 0.8, "uses": 5}
]
limit = 3
assert retrieval.rank_by_overlap(items, context, searchable, tiebreak, limit) == items

print('OK')
