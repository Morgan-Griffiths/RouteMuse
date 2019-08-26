import json


def build_data():
    fields = {
        'grade': {
            0: 'blue',
            1: 'pink',
            2: 'purple',
            3: 'green',
            4: 'yellow',
            5: 'orange',
            6: 'red',
            7: 'grey',
            8: 'black',
            9: 'white'
        },
        'terrain_type': {
            0: "roof",
            1: "overhung",
            2: "vertical",
            3: "slab"
        },
        'locations': {0: "Red South",
                      1: "Red Roof",
                      2: "Red North",
                      3: "Split Seam",
                      4: "Titan North",
                      5: "Slab Roof",
                      6: "Titan South",
                      7: "Liberty Face",
                      8: "45 Degree",
                      9: "20 Degree",
                      10: "Great Roof",
                      11: "Horizontal Barrel",
                      12: "Shield",
                      13: "Accordion",
                      14: "Warped Slab",
                      15: "Topout Cube",
                      16: "Topout North",
                      17: "Wave Wall",
                      18: "Comp Slab",
                      19: "Vertical Barrel"
                      },
        'intra_difficulty': {
            0: 1,
            1: 2,
            2: 3
        },
        'risk': {
            0: 1,
            1: 2,
            2: 3,
            3: 4,
            4: 5
        },
        'intensity': {
            0: 1,
            1: 2,
            2: 3,
            3: 4,
            4: 5
        },
        'complexity': {
            0: 1,
            1: 2,
            2: 3,
            3: 4,
            4: 5

        },
        'height_friendly': {
            0: 'Average',
            1: 'Short',
            2: 'Tall'
        },
        'start_location': {
            0: 'Middle',
            1: 'Left',
            2: 'Right',
            3: 'Left-middle',
            4: 'Right-middle'
        },
        'finish_location': {
            0: 'Middle',
            1: 'Left',
            2: 'Right',
            3: 'Left-middle',
            4: 'Right-middle'
        },
        'style': {
            0: "finger strength",
            1: "simple",
            2: "body strength",
            3: "complex",
            4: "technical",
            5: "cryptic",
            6: "powerful",
            7: "sustained",
            8: "crux",
            9: "progressive",
            10: "layback",
            11: "balance",
            12: "dynamic",
            13: "traverse",
            14: "Compression",
            15: "thugish",
            16: "Sequential",
            17: "Stemming"
        },
        'techniques': {
            0: "toe hooking",
            1: "heel hooking",
            2: "mantle",
            3: 'gaston',
            4: 'twist',
            5: 'cross',
            6: 'campus',
            7: 'layback',
            8: 'stemming',
            9: 'lock off',
            10: 'Dyno',
            11: 'high step',
            12: 'flagging',
            13: 'Bumping'
        }}
    json_data = json.dumps(fields)
    return json_data
