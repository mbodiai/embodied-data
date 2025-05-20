import pytest
from embdata.sample import Sample
from embdata.list import List

import pytest
from embdata.sample import Sample
from embdata.list import List

# Define the sample models
class MyModel(Sample):
    a: int
    b: str
    c: bool
    d: float

class MySimpleModel(Sample):
    a: int
    b: str

class MyComplexModel(Sample):
    x: int
    y: dict[str, list[int]]
    z: list[MySimpleModel]

# Test Data
my_model_1 = MyModel(a=1, b="hello", c=True, d=3.14)
my_model_2 = MyModel(a=2, b="world", c=False, d=6.28)
my_model_3 = MyModel(a=3, b="test", c=True, d=1.11)

my_simple_model_1 = MySimpleModel(a=1, b="simple1")
my_simple_model_2 = MySimpleModel(a=2, b="simple2")

my_complex_model_1 = MyComplexModel(x=10, y={"key1": [1, 2, 3], "key2": [4, 5]}, z=[my_simple_model_1, my_simple_model_2])

# Fixtures for MyModel
@pytest.fixture
def my_model_list_1():
    return List[MyModel]([my_model_1])

@pytest.fixture
def my_model_list_2():
    return List[MyModel]([my_model_2])

@pytest.fixture
def my_model_list_3():
    return List[MyModel]([my_model_1, my_model_2])

# Fixture for MySimpleModel
@pytest.fixture
def my_simple_model_list():
    return List[MySimpleModel]([my_simple_model_1, my_simple_model_2])

# Fixture for MyComplexModel
@pytest.fixture
def my_complex_model_list():
    return List[MyComplexModel]([my_complex_model_1])


def test_list_initialization(my_model_list_1):
    """Test initializing the List with items."""
    assert len(my_model_list_1) == 1
    assert my_model_list_1[0] == my_model_1
    assert isinstance(my_model_list_1, List)

def test_list_append(my_model_list_1):
    """Test appending items to the list."""
    my_model_list_1.append(my_model_2)
    assert len(my_model_list_1) == 2
    assert my_model_list_1[1] == my_model_2

def test_list_extend(my_model_list_1):
    """Test extending the list with multiple items."""
    my_model_list_1.extend([my_model_2, my_model_3])
    assert len(my_model_list_1) == 3
    assert my_model_list_1[1] == my_model_2
    assert my_model_list_1[2] == my_model_3

def test_list_pop(my_model_list_1):
    """Test popping items from the list."""
    my_model_list_1.append(my_model_2)
    popped_item = my_model_list_1.pop(1)
    assert popped_item == my_model_2
    assert len(my_model_list_1) == 1

def test_list_remove(my_model_list_3):
    """Test removing an item from the list."""
    my_model_list_3.remove(my_model_1)
    assert len(my_model_list_3) == 1
    assert my_model_list_3[0] == my_model_2

def test_list_clear(my_model_list_1):
    """Test clearing all items from the list."""
    my_model_list_1.clear()
    assert len(my_model_list_1) == 0

def test_list_reverse(my_model_list_3):
    """Test reversing the list."""
    my_model_list_3.reverse()
    assert my_model_list_3[0] == my_model_2
    assert my_model_list_3[1] == my_model_1

def test_list_sort(my_model_list_3):
    """Test sorting the list by a specific key."""
    my_model_list_3.extend([my_model_3])
    my_model_list_3.sort(key=lambda x: x.a)
    assert my_model_list_3[0] == my_model_1
    assert my_model_list_3[1] == my_model_2
    assert my_model_list_3[2] == my_model_3

def test_list_serialize(my_model_list_3):
    """Test serializing the list to a list of dictionaries."""
    serialized = my_model_list_3.serialize()
    assert isinstance(serialized, list)
    assert serialized[0] == my_model_1.model_dump()
    assert serialized[1] == my_model_2.model_dump()

def test_list_model_validate():
    """Test validating a list of MyModel items."""
    validated_list = List[MyModel].model_validate([my_model_1.model_dump(), my_model_2.model_dump(), my_model_3.model_dump()])
    assert len(validated_list) == 3
    assert validated_list[0] == my_model_1
    assert validated_list[1] == my_model_2
    assert validated_list[2] == my_model_3

def test_list_model_json_schema():
    """Test generating JSON schema for the list."""
    schema = List[MyModel].model_json_schema()
    
    # Ensure that the top-level schema is an array type
    assert schema["type"] == "array"
    assert "items" in schema
    
    # Check that the definition for MyModel exists in $defs
    assert "$defs" in schema
    assert "MyModel" in schema["$defs"]
    
    # Check that MyModel contains the expected properties
    model_schema = schema["$defs"]["MyModel"]
    assert "properties" in model_schema
    assert "a" in model_schema["properties"]
    assert "b" in model_schema["properties"]
    assert "c" in model_schema["properties"]
    assert "d" in model_schema["properties"]


def test_list_schema():
    """Test generating JSON schema for the list."""
    schema = List[MyModel].schema()
    print(f"Schema: {schema}")  
    
    # Ensure the top-level schema is an array type
    assert schema["type"] == "array"
    assert "items" in schema

    # Check that 'properties' exist in the schema for each item
    assert "properties" in schema["items"]
    assert "a" in schema["items"]["properties"]
    assert "b" in schema["items"]["properties"]
    assert "c" in schema["items"]["properties"]
    assert "d" in schema["items"]["properties"]

# New test for complex model
def test_complex_model_serialize(my_complex_model_list):
    """Test serializing the complex model list."""
    serialized = my_complex_model_list.serialize()
    assert isinstance(serialized, list)
    assert serialized[0] == my_complex_model_1.model_dump()

def test_complex_model_model_validate():
    """Test validating a list of MyComplexModel items."""
    validated_list = List[MyComplexModel].model_validate([my_complex_model_1.model_dump()])
    assert len(validated_list) == 1
    assert validated_list[0] == my_complex_model_1

def test_complex_model_model_json_schema():
    """Test generating JSON schema for the complex model list."""
    schema = List[MyComplexModel].model_json_schema()
    print(f"Schema: {schema}")
    # Ensure that the top-level schema is an array type
    assert schema["type"] == "array"
    assert "items" in schema
    
    # Check that the definition for MyComplexModel exists in $defs
    assert "$defs" in schema
    assert "MyComplexModel" in schema["$defs"]
    
    # Check that MyComplexModel contains the expected properties
    model_schema = schema["$defs"]["MyComplexModel"]
    assert "properties" in model_schema
    assert "x" in model_schema["properties"]
    assert "y" in model_schema["properties"]
    assert "z" in model_schema["properties"]

    # Check that the 'y' property uses 'additionalProperties'
    assert "y" in model_schema["properties"]
    assert "type" in model_schema["properties"]["y"]
    assert model_schema["properties"]["y"]["type"] == "object"
    assert "additionalProperties" in model_schema["properties"]["y"]
    assert model_schema["properties"]["y"]["additionalProperties"]["type"] == "array"
    assert model_schema["properties"]["y"]["additionalProperties"]["items"]["type"] == "integer"

    # Check that the 'z' property is an array of MySimpleModel items
    assert "z" in model_schema["properties"]
    assert "type" in model_schema["properties"]["z"]
    assert model_schema["properties"]["z"]["type"] == "array"
    assert "items" in model_schema["properties"]["z"]
    assert "$ref" in model_schema["properties"]["z"]["items"]
    assert model_schema["properties"]["z"]["items"]["$ref"] == "#/$defs/MySimpleModel"

def test_complex_model_schema():
    """Test generating JSON schema for the complex model list."""
    schema = List[MyComplexModel].schema()
    print(f"Schema: {schema}")
    
    # Ensure the top-level schema is an array type
    assert schema["type"] == "array"
    assert "items" in schema

    # Check that 'properties' exist in the schema for each item
    assert "properties" in schema["items"]
    assert "x" in schema["items"]["properties"]
    assert "y" in schema["items"]["properties"]
    assert "z" in schema["items"]["properties"]

    # Check that the 'y' property uses 'additionalProperties'
    assert "y" in schema["items"]["properties"]
    assert "type" in schema["items"]["properties"]["y"]
    assert schema["items"]["properties"]["y"]["type"] == "object"
    assert "additionalProperties" in schema["items"]["properties"]["y"]
    assert schema["items"]["properties"]["y"]["additionalProperties"]["type"] == "array"
    assert schema["items"]["properties"]["y"]["additionalProperties"]["items"]["type"] == "integer"

    # Check that the 'z' property is an array of MySimpleModel items
    assert "z" in schema["items"]["properties"]
    assert "type" in schema["items"]["properties"]["z"]
    assert schema["items"]["properties"]["z"]["type"] == "array"
    assert "items" in schema["items"]["properties"]["z"]

    # Check the properties of MySimpleModel are embedded directly (since $ref may not be used)
    z_items = schema["items"]["properties"]["z"]["items"]
    assert "properties" in z_items
    assert "a" in z_items["properties"]
    assert "b" in z_items["properties"]
    assert z_items["properties"]["a"]["type"] == "integer"
    assert z_items["properties"]["b"]["type"] == "string"

def test_complex_sort(my_complex_model_list):
    """Test sorting the complex model list."""
    my_complex_model_list.sort(key=lambda x: x.x)
    assert my_complex_model_list[0] == my_complex_model_1

def test_complex_append(my_complex_model_list):
    """Test appending a new item to the complex model list."""
    new_model = MyComplexModel(x=20, y={"key3": [6, 7, 8]}, z=[my_simple_model_1])
    my_complex_model_list.append(new_model)
    assert len(my_complex_model_list) == 2
    assert my_complex_model_list[1] == new_model

def test_complex_extend(my_complex_model_list):
    """Test extending the complex model list with multiple items."""
    new_model_1 = MyComplexModel(x=20, y={"key3": [6, 7, 8]}, z=[my_simple_model_1])
    new_model_2 = MyComplexModel(x=30, y={"key4": [9, 10]}, z=[my_simple_model_2])
    my_complex_model_list.extend([new_model_1, new_model_2])
    assert len(my_complex_model_list) == 3
    assert my_complex_model_list[1] == new_model_1
    assert my_complex_model_list[2] == new_model_2

def test_complex_pop(my_complex_model_list):
    """Test popping an item from the complex model list."""
    new_model = MyComplexModel(x=20, y={"key3": [6, 7, 8]}, z=[my_simple_model_1])
    my_complex_model_list.append(new_model)

    popped_item = my_complex_model_list.pop(0)
    
    assert popped_item == my_complex_model_1  
    assert len(my_complex_model_list) == 1  
