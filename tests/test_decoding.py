import torch

from models.video_captioning_moe import VideoCaptioningMoE
from train.eval import greedy_decode, beam_search_decode


def test_greedy_and_beam_shapes():
    model = VideoCaptioningMoE(vocab_size=100)
    video = torch.rand(1, 16, 3, 224, 224)
    g = greedy_decode(model, video, max_len=8)
    b = beam_search_decode(model, video, max_len=8, beam_size=3, length_penalty=0.7)
    assert g.shape == (1, 8)
    assert b.shape == (1, 8)
