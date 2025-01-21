import pytest
import pandas as pd
import tempfile
from pathlib import Path
import json
import requests
from unittest.mock import Mock, patch
from arthorian_quest.query import QueryArthor


# Mock data and fixtures
@pytest.fixture
def mock_response():
    """Mock successful API response"""
    return {
        'data': [
            {'arthor.rank': 1, 'arthor.index': 1, 'smiles': 'CC', 'identifier': 'id1', 'arthor.source': 'src1'},
            {'arthor.rank': 2, 'arthor.index': 2, 'smiles': 'CCC', 'identifier': 'id2', 'arthor.source': 'src2'}
        ],
        'recordsTotal': 2,
        'recordsFiltered': 2,
        'hasMore': False
    }


@pytest.fixture
def mock_empty_response():
    """Mock API response with no matches"""
    return {
        'data': [],
        'recordsTotal': 0,
        'recordsFiltered': 0,
        'hasMore': False
    }

@pytest.fixture
def temp_cache_dir():
    """Create temporary directory for cache testing"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


# Basic functionality tests
def test_batch_retrieve_basic(mock_response):
    """Test basic batch retrieval functionality"""
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.status_code = 200

        qa = QueryArthor()
        queries = ['CC', 'CCC']
        result = qa.batch_retrieve(queries, ['test_db'])

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert mock_get.call_count == len(queries)


def test_batch_retrieve_with_cache(temp_cache_dir, mock_response):
    """Test batch retrieval with caching enabled"""
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.status_code = 200

        qa = QueryArthor(cache_dir=temp_cache_dir)
        queries = ['CC', 'CCC']
        result = qa.batch_retrieve(queries, ['test_db'])

        # Check cache files were created
        assert (temp_cache_dir / "batch_progress.json").exists()
        assert (temp_cache_dir / "combined_results.csv").exists()

        # Check progress file content
        with open(temp_cache_dir / "batch_progress.json") as f:
            progress = json.load(f)
            assert all(progress[q] for q in queries)


def test_batch_retrieve_error_handling(mock_response):
    """Test error handling during batch retrieval"""
    with patch('requests.get') as mock_get:
        # First query succeeds, second fails
        def side_effect(*args):
            query_params = args[1].get('query')

            # Exact match for 'CC' only
            if query_params == 'CC':
                mock = Mock()
                mock.json.return_value = mock_response
                mock.status_code = 200
                return mock
            else:
                raise requests.exceptions.RequestException("Test error")

        mock_get.side_effect = side_effect

        qa = QueryArthor()
        queries = ['CC', 'CCC']

        # Test with continue_on_error=True
        result = qa.batch_retrieve(queries, ['test_db'], continue_on_error=True)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

        # Test with continue_on_error=False
        with pytest.raises(Exception):
            qa.batch_retrieve(queries, ['test_db'], continue_on_error=False)


def test_batch_statistics(temp_cache_dir, mock_response):
    """Test batch statistics functionality"""
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.status_code = 200

        qa = QueryArthor(cache_dir=temp_cache_dir)
        queries = ['CC', 'CCC']
        qa.batch_retrieve(queries, ['test_db'])

        stats = qa.get_batch_statistics()
        assert stats['total_processed'] == len(queries)
        assert stats['successful'] == len(queries)
        assert stats['failed'] == 0


def test_resume_interrupted_batch(temp_cache_dir, mock_response):
    """Test resuming an interrupted batch process"""
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.status_code = 200

        # Create progress file with one completed query
        progress = {'CC': True, 'CCC': False}
        (temp_cache_dir / "batch_progress.json").write_text(json.dumps(progress))

        qa = QueryArthor(cache_dir=temp_cache_dir)
        queries = ['CC', 'CCC']
        result = qa.batch_retrieve(queries, ['test_db'])

        # Should only process the failed query
        assert mock_get.call_count == 1
        assert isinstance(result, pd.DataFrame)


def test_empty_results_handling(mock_empty_response):
    """Test handling of queries that return no results"""
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = mock_empty_response
        mock_get.return_value.status_code = 200

        qa = QueryArthor()
        queries = ['CC']
        result = qa.batch_retrieve(queries, ['test_db'])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# Edge cases and validation tests
def test_invalid_cache_dir():
    """Test behavior with invalid cache directory"""
    with pytest.raises(Exception):
        QueryArthor(cache_dir='/nonexistent/path/that/cannot/be/created')


def test_empty_queries_list():
    """Test behavior with empty queries list"""
    qa = QueryArthor()
    result = qa.batch_retrieve([], ['test_db'])
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_statistics_no_cache():
    """Test statistics behavior when no cache directory is set"""
    qa = QueryArthor()
    stats = qa.get_batch_statistics()
    assert "error" in stats


def test_invalid_search_type():
    """Test behavior with invalid search type"""
    qa = QueryArthor()
    with pytest.raises(Exception):
        qa.batch_retrieve('CC', ['test_db'], search_type='InvalidType')