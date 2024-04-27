import os


def init_nuclides():
    with open(
        os.path.join(
            os.path.dirname(__file__),
            "unique_nuclides.txt"
        ),
        "r"
    ) as f:
        content = list(map(lambda x: x.strip(), f.readlines())) 
    
    return content