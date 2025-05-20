from embdata.sense.depth import Depth
from embdata.sense.world import World
from embdata.sense.world_object import WorldObject, Collection
from embdata.coordinate import PixelCoords, BBox2D
from embdata.sense.image import Image
from embdata.sense.camera import Camera, Distortion, Intrinsics
from embdata.coordinate import Coordinate, Point, Pose6D, Mask, Plane
from embdata.sense.aruco import ArucoParams, Aruco
from embdata.ndarray import ndarray
import numpy as np
import pytest
from embdata.coordinate import Pose
from importlib.resources import files


DEPTH_IMAGE_PATH = files("embdata") / "resources/depth_image.png"
COLOR_IMAGE_PATH = files("embdata") / "resources/color_image.png"


WORLD_POSE = Pose(x=0.0, y=-0.2032, z=0.0, roll=0, pitch=0.0, yaw=90)


RGB_IMAGE = Image(path=COLOR_IMAGE_PATH,
                      encoding="png",
                      mode="RGB")


ARUCO_PARAMS = ArucoParams(marker_size=0.1,
                           world_pose=WORLD_POSE)


CAMERA = Camera(
    intrinsic=Intrinsics(fx=911, fy=911, cx=653, cy=371),
    distortion=Distortion(k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0),
    depth_scale=0.001,
)


DEPTH_IMAGE = Depth(path=DEPTH_IMAGE_PATH,
                        encoding="png",
                        mode="I",
                        size=(1280, 720),
                        camera=CAMERA,
                        unit="mm",
                        rgb=RGB_IMAGE)


ARUCO = Aruco(image=RGB_IMAGE,
              camera=CAMERA,
              params=ARUCO_PARAMS,
              depth=DEPTH_IMAGE)



ARUCO_OBJECT = WorldObject(
    name='aruco',
    bbox_2d=BBox2D(
        x1=918.0,
        y1=537.0,
        x2=846.0,
        y2=517.0,
        confidence=None,
        label=None
    ),
    pixel_coords=PixelCoords(u=881, v=527),
    mask=None,
    image=None,
    pose=Pose(
        x=-1.3268537045390063,
        y=-0.08928884992217898,
        z=-0.3512184400694646,
        roll=-1.6683278893587614,
        pitch=-0.00528035502267743,
        yaw=1.7368956029586198
    ),
    bbox_3d=None,
    volume=None,
    points=None,
    xyz_min=None,
    xyz_max=None,
    pca_vectors=None
)


@pytest.fixture
def test_world() -> World:
    world = World()
    world.image = RGB_IMAGE
    world.depth = DEPTH_IMAGE
    world.camera = CAMERA
    world.aruco = ARUCO
    return world


@pytest.fixture
def remote_control_object() -> WorldObject:
    """Fixture for the REMOTE_CONTROL_OBJECT."""
    remote_control_object = WorldObject(
        name='remote control',
        bbox_2d=BBox2D(
            x1=988.9449462890625,
            y1=574.7791137695312,
            x2=1097.5645751953125,
            y2=687.1047973632812,
            confidence=None,
            label=None
        ),
        pixel_coords=PixelCoords(u=1043, v=630),
        mask=None,
        image=None,
        pose=Pose6D(
            x=0.36212678375411633,
            y=0.24408671789242592,
            z=0.8654999999999999,
            roll=0.0,
            pitch=0.0,
            yaw=0.0
        ),
        bbox_3d=None,
        volume=None,
        points=None,
        xyz_min=ndarray([0.333430296377607, 0.22716136114160265, 0.764]),
        xyz_max=ndarray([0.3908232711306257, 0.26101207464324916, 0.967]),
        pca_vectors=ndarray([[     0.1778,    -0.98144,    0.071856],
           [   -0.12053,   -0.094189,    -0.98823],
           [    0.97666,     0.16705,    -0.13504]])
    )
    remote_control_object.pose.set_reference_frame(reference_frame="camera")
    remote_control_object.pose.set_info("origin", Pose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0))
    return remote_control_object


@pytest.fixture
def marker_object() -> WorldObject:
    """Fixture for the MARKER_OBJECT."""
    marker_object = WorldObject(
        name='marker',
        bbox_2d=BBox2D(
            x1=823.9132080078125,
            y1=633.4031372070312,
            x2=848.3844604492188,
            y2=706.2813110351562,
            confidence=None,
            label=None
        ),
        pixel_coords=PixelCoords(u=836, v=669),
        mask=None,
        image=None,
        pose=Pose6D(
            x=0.15699794182217341,
            y=0.2573982985729967,
            z=0.7869999999999999,
            roll=0.0,
            pitch=0.0,
            yaw=0.0
        ),
        bbox_3d=None,
        volume=None,
        points=None,
        xyz_min=ndarray([0.14198051591657518, 0.24873545554335896, 0.731]),
        xyz_max=ndarray([0.17201536772777168, 0.26606114160263444, 0.843]),
        pca_vectors=ndarray([[    0.19644,    -0.98049,   0.0076478],
           [   -0.10805,   -0.029399,    -0.99371],
           [    0.97454,     0.19438,    -0.11172]])
    )
    marker_object.pose.set_reference_frame(reference_frame="camera")
    marker_object.pose.set_info("origin", Pose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0))
    return marker_object


@pytest.fixture
def setup_collection() -> Collection[WorldObject]:
    """Fixture to create a collection of WorldObjects."""
    return Collection[WorldObject]()


def test_append_and_retrieve_by_key(setup_collection):
    """Test appending items and retrieving them by key."""
    collection = setup_collection

    # Create and append objects
    obj1 = WorldObject(name="object1", pixel_coords=PixelCoords(u=100, v=200))
    obj2 = WorldObject(name="object2", pixel_coords=PixelCoords(u=300, v=400))
    obj3 = WorldObject(name="object1", pixel_coords=PixelCoords(u=500, v=600))  # Duplicate key

    collection.append(obj1)
    collection.append(obj2)
    collection.append(obj3)

    # Assert that objects are retrievable by key
    assert collection["object1"] == obj1  # Gets the first "object1"
    assert collection.getall("object1") == [obj1, obj3]  # Should return a list of all "object1"
    assert collection["object2"] == obj2


def test_retrieve_by_index(setup_collection):
    """Test retrieving items by index."""
    collection = setup_collection

    # Create and append objects
    obj1 = WorldObject(name="object1", pixel_coords=PixelCoords(u=100, v=200))
    obj2 = WorldObject(name="object2", pixel_coords=PixelCoords(u=300, v=400))

    collection.append(obj1)
    collection.append(obj2)

    # Assert retrieval by index
    assert collection[0] == obj1
    assert collection[1] == obj2


def test_iteration(setup_collection):
    """Test iteration over the collection."""
    collection = setup_collection

    # Create and append objects
    obj1 = WorldObject(name="object1", pixel_coords=PixelCoords(u=100, v=200))
    obj2 = WorldObject(name="object2", pixel_coords=PixelCoords(u=300, v=400))

    collection.append(obj1)
    collection.append(obj2)

    # Assert that iteration yields all objects
    items = list(iter(collection))
    assert items == [obj1, obj2]


def test_multiple_iteration(setup_collection):
    """Test iteration over the collection."""
    collection = setup_collection

    # Create and append objects
    obj1 = WorldObject(name="object1", pixel_coords=PixelCoords(u=100, v=200))
    obj2 = WorldObject(name="object1", pixel_coords=PixelCoords(u=100, v=200))
    obj3 = WorldObject(name="object1", pixel_coords=PixelCoords(u=100, v=200))
    obj4 = WorldObject(name="object2", pixel_coords=PixelCoords(u=300, v=400))

    collection.append(obj1)
    collection.append(obj2)
    collection.append(obj3)
    collection.append(obj4)

    # Assert that iteration yields all objects
    items = list(iter(collection))
    assert items == [obj1, obj2, obj3, obj4]


def test_length(setup_collection):
    """Test the length calculation of the collection."""
    collection = setup_collection

    # Create and append objects
    obj1 = WorldObject(name="object1", pixel_coords=PixelCoords(u=100, v=200))
    obj2 = WorldObject(name="object2", pixel_coords=PixelCoords(u=300, v=400))

    collection.append(obj1)
    collection.append(obj2)

    # Assert that length matches the number of objects
    assert len(collection) == 2


def test_concatenation():
    """Test concatenating multiple collections."""
    collection1 = Collection[WorldObject]()
    collection2 = Collection[WorldObject]()

    obj1 = WorldObject(name="object1", pixel_coords=PixelCoords(u=100, v=200))
    obj2 = WorldObject(name="object2", pixel_coords=PixelCoords(u=300, v=400))
    obj3 = WorldObject(name="object3", pixel_coords=PixelCoords(u=500, v=600))

    collection1.append(obj1)
    collection2.append(obj2)
    collection2.append(obj3)

    concatenated = Collection.concat([collection1, collection2])
    
    # Assert all objects are in the concatenated collection
    assert len(concatenated) == 3
    assert concatenated.getall("object1") == [obj1]
    assert concatenated.getall("object2") == [obj2]
    assert concatenated.getall("object3") == [obj3]


@pytest.mark.network
def test_fastapi(test_world: World):
    from fastapi import FastAPI
    from httpx import Client
    from embdata.utils.network_utils import get_open_port
    from time import sleep
    import uvicorn
    
    app = FastAPI()
    
    @app.post("/test")
    async def test(d: World) -> World:
        assert isinstance(d, World)
        assert d.objects == test_world.objects
        return world
    port = get_open_port()
    
    import threading
    thread = threading.Thread(target=uvicorn.run, args=(app,), kwargs={"host": "localhost", "port": port}, daemon=True)
    thread.start()
    sleep(5)
    client = Client()
    world = test_world
    response = client.post(f"http://localhost:{port}/test", json=world.model_dump(mode="json"))

    assert response.status_code == 200
    world_resp = World(**response.json())
    assert np.allclose(world_resp.depth.array, world.depth.array)
    assert world_resp.camera == world.camera
    assert world_resp.objects == world.objects

    thread.join(timeout=10)  # Add a timeout to prevent hanging

        

if __name__ == "__main__":
    pytest.main([__file__, "-vv"])