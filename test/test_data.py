import json

def build_data():
    fields = {
    'grade': {
        0:'blue',
        1:'pink',
        2:'purple',
        3:'green',
        4:'yellow',
        5:'orange',
        6:'red',
        7:'grey',
        8:'black',
        9:'white'
    },
    'locations': {0:"Red South",
                    1:"Red Roof",
                    2:"Red North",
                    3:"Split Seam",
                    4:"Titan North",
                    5:"Slab Roof",
                    6:"Titan South",
                    7:"Liberty Face",
                    8:"45 Degree",
                    9:"20 Degree",
                    10:"Great Roof",
                    11:"Horizontal Barrel",
                    12:"Shield",
                    13:"Accordion",
                    14:"Warped Slab",
                    15:"Topout Cube",
                    16:"Topout North",
                    17:"Wave Wall",
                    18:"Comp Slab",
                    19:"Vertical Barrel"
                },
    'intra_difficulty':{
                    0:1,
                    1:2,
                    2:3
                },
    'risk':{
                    0:1,
                    1:2,
                    2:3,
                    3:4,
                    4:5
                },
    'intensity':{
                    0:1,
                    1:2,
                    2:3,
                    3:4,
                    4:5
                },
    'complexity':{
                    0:1,
                    1:2,
                    2:3,
                    3:4,
                    4:5

                },
    'height_friendly':{
                0:'Average',
                1:'Short',
                2:'Tall'
            },
    'start_location':{
        0:'Middle',
        1:'Left',
        2:'Right',
        3:'Left-middle',
        4:'Right-middle'
    },
    'finish_location':{
        0:'Middle',
        1:'Left',
        2:'Right',
        3:'Left-middle',
        4:'Right-middle'
    },
    'style':{
            0:"finger strength",
            1:"simple",
            2:"body strength",
            3:"complex",
            4:"technical",
            5:"cryptic",
            6:"powerful",
            7:"sustained",
            8:"crux",
            9:"progressive",
            10:"layback",
            11:"balance",
            12:"dynamic",
            13:"traverse",
            14:"Compression",
            15:"thugish",
            16:"Sequential",
            17:"Stemming"
    },
    'techniques':{
            0:"footwork",
            1:"heel hooking",
            2:"static movement",
            3:"dynamic movement",
            4:"toe hooking",
            5:"direction of pull",
            6:"sequence",
            7:"mantle",
            8:"smearing",
            9:"pain",
            10:"body position",
            11:"campusing",
            12:"drop knee",
            13:"hand position",
            14:"layback",
            15:"roof footwork",
            16:"paddle",
            17:"compression",
            18:"stem",
            19:"lockoff",
            20:"precision",
            21:"knee bar",
            22:"Getting to the top",
            23:"Traversing",
            24:"Balance",
            25:"Smearing",
            26:"Pressing"
    },
    'hold_sets':{
                0:"egrips_hueco_crimps",
            1:"egrips_hueco_buckets",
            2:"kilter_finger_buckets",
            3:"rustic_flowers",
            4:"kilter_crimps",
            5:"teknik_plats",
            6:"egrips_discs",
            7:"kingdom_jousting_jugs",
            8:"flathold_sloper_volumes",
            9:"kingdom_dragonballs",
            10:"kingdom_flakes",
            11:"soill_innies",
            12:"egrips_huecos",
            13:"rockcandy_blockus"
    }}
    json_data = json.dumps(fields)
    return json_data
