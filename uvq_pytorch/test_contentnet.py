import numpy as np
import pytest
from utils.contentnet import ContentNetInference


class TestContentNetInference:
    
    @pytest.fixture
    def contentnet_inference(self):
        """Create ContentNetInference instance without loading the actual model."""
        return ContentNetInference(pretrained=False, eval_mode=False)
    
    def test_label_probabilities_to_text_top_1(self, contentnet_inference):
        """Test label_probabilities_to_text with top_n=1."""
        # Create mock probabilities with highest at index 0 (Game)
        label_probs = np.zeros(3862)
        label_probs[0] = 0.9  # Game
        label_probs[1] = 0.05  # Video game
        label_probs[2] = 0.03  # Vehicle
        
        predicted, probs, indices = contentnet_inference.label_probabilities_to_text(
            label_probs, top_n=1
        )
        
        assert len(predicted) == 1
        assert len(probs) == 1
        assert len(indices) == 1
        assert predicted[0] == "Game"
        assert probs[0] == 0.9
        assert indices[0] == 0
    
    def test_label_probabilities_to_text_top_3(self, contentnet_inference):
        """Test label_probabilities_to_text with top_n=3."""
        # Create mock probabilities
        label_probs = np.zeros(3862)
        label_probs[0] = 0.5   # Game
        label_probs[1] = 0.3   # Video game  
        label_probs[2] = 0.15  # Vehicle
        label_probs[3] = 0.05  # Concert
        
        predicted, probs, indices = contentnet_inference.label_probabilities_to_text(
            label_probs, top_n=3
        )
        
        assert len(predicted) == 3
        assert len(probs) == 3
        assert len(indices) == 3
        
        # Check order (should be sorted by probability descending)
        assert predicted[0] == "Game"
        assert predicted[1] == "Video game"
        assert predicted[2] == "Vehicle"
        
        assert probs[0] == 0.5
        assert probs[1] == 0.3
        assert probs[2] == 0.15
        
        assert indices[0] == 0
        assert indices[1] == 1
        assert indices[2] == 2
    
    def test_label_probabilities_to_text_random_order(self, contentnet_inference):
        """Test that results are properly sorted by probability regardless of input order."""
        label_probs = np.zeros(3862)
        label_probs[7] = 0.1   # Car (lower probability)
        label_probs[8] = 0.9   # Dance (higher probability)  
        label_probs[6] = 0.5   # Performance art (middle probability)
        
        predicted, probs, indices = contentnet_inference.label_probabilities_to_text(
            label_probs, top_n=3
        )
        
        # Should be sorted by probability descending
        assert predicted[0] == "Dance"
        assert predicted[1] == "Performance art"
        assert predicted[2] == "Car"
        
        assert probs[0] == 0.9
        assert probs[1] == 0.5
        assert probs[2] == 0.1
        
        assert indices[0] == 8
        assert indices[1] == 6
        assert indices[2] == 7
    
    def test_label_probabilities_to_text_edge_case_top_n_larger_than_nonzero(self, contentnet_inference):
        """Test when top_n is larger than number of non-zero probabilities."""
        label_probs = np.zeros(3862)
        label_probs[0] = 0.7  # Game
        label_probs[1] = 0.3  # Video game
        # Only 2 non-zero values but asking for top 5
        
        predicted, probs, indices = contentnet_inference.label_probabilities_to_text(
            label_probs, top_n=5
        )
        
        # Should return 5 results (including zeros)
        assert len(predicted) == 5
        assert len(probs) == 5
        assert len(indices) == 5
        
        # First two should be the non-zero ones
        assert predicted[0] == "Game"
        assert predicted[1] == "Video game"
        assert probs[0] == 0.7
        assert probs[1] == 0.3
        
        # Remaining should have probability 0.0
        assert probs[2] == 0.0
        assert probs[3] == 0.0
        assert probs[4] == 0.0
