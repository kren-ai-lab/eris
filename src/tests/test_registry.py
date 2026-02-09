from morpheo.core.registry import list_encoders

def test_registry_empty_ok():
    assert isinstance(list_encoders(), list)
