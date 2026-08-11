import unittest

from mohim_dali.dali import DaliTrack, extract_plain_lyrics, filter_tracks


class FakeEntry:
    def __init__(self):
        self.annotations = {
            "type": "horizontal",
            "annot": {
                "lines": [
                    {"text": "First line", "time": [0.0, 1.0]},
                    {"text": "Second line", "time": [1.0, 2.0]},
                ]
            },
        }


class DaliTests(unittest.TestCase):
    def test_extract_plain_lyrics_discards_timestamps(self):
        self.assertEqual(extract_plain_lyrics(FakeEntry()), "First line\nSecond line")

    def test_filter_english_pop_with_lyrics(self):
        entry = FakeEntry()
        tracks = [
            DaliTrack("1", "A", "Pop", ("Pop",), "English", "lyrics", None, None, entry),
            DaliTrack("2", "B", "Rock", ("Rock",), "English", "lyrics", None, None, entry),
            DaliTrack("3", "C", "No lyrics", ("Pop",), "ENG", "", None, None, entry),
        ]
        self.assertEqual([track.dali_id for track in filter_tracks(tracks)], ["1"])


if __name__ == "__main__":
    unittest.main()
