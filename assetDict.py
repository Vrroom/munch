from drawTools import *

ASSET_DICT = \
"""
dict(
    wall=[
        dict(
            obj=load_blend_model('./assets/walls.blend', 'wall_1_by_1_light'), 
            aspect=(1, 1),
            props=set(['light'])
        ),
        dict(
            obj=load_blend_model('./assets/walls.blend', 'wall_1_by_2'), 
            aspect=(1, 2),
            props=set(['dark'])
        ),
        dict(
            obj=load_blend_model('./assets/walls.blend', 'wall_0-5_by_2'), 
            aspect=(1, 4),
            props=set(['dark'])
        ),
        dict(
            obj=load_blend_model('./assets/walls.blend', 'wall_0-5_by_2_light'), 
            aspect=(1, 4),
            props=set(['light'])
        ),
        dict(
            obj=load_blend_model('./assets/walls.blend', 'wall_1_by_2_light'), 
            aspect=(1, 2),
            props=set(['light'])
        ),
        dict(
            obj=load_blend_model('./assets/walls.blend', 'wall_1_by_3'), 
            aspect=(1, 3),
            props=set(['dark'])
        ),
        dict(
            obj=load_blend_model('./assets/walls.blend', 'wall_1_by_3_light'), 
            aspect=(1, 3),
            props=set(['light'])
        ),
        dict(
            obj=load_blend_model('./assets/walls.blend', 'wall_2_by_1'), 
            aspect=(2, 1),
            props=set(['dark'])
        ),
        dict(
            obj=load_blend_model('./assets/walls.blend', 'wall_2_by_1_light'), 
            aspect=(2, 1),
            props=set(['light'])
        ),
        dict(
            obj=load_blend_model('./assets/walls.blend', 'wall_2_by_3'), 
            aspect=(2, 3),
            props=set(['dark'])
        ),
        dict(
            obj=load_blend_model('./assets/walls.blend', 'wall_2_by_3_light'), 
            aspect=(2, 3),
            props=set(['light'])
        ),
    ],
    wall_window=[
        dict(
            obj=load_blend_model('./assets/walls.blend', 'wall_window_center_2_by_3'), 
            aspect=(2, 3),
            props=set(['dark'])
        ),
        dict(
            obj=load_blend_model('./assets/walls.blend', 'wall_window_center_2_by_3_light'), 
            aspect=(2, 3),
            props=set(['light'])
        ),
        dict(
            obj=load_blend_model('./assets/walls.blend', 'wall_window_center_2_by_3_light_2'), 
            aspect=(2, 3),
            props=set(['light'])
        ),
        dict(
            obj=load_blend_model('./assets/walls.blend', 'wall_window_inset_center_2_by_3'), 
            aspect=(2, 3),
            props=set(['dark'])
        ),
        dict(
            obj=load_blend_model('./assets/walls.blend', 'wall_window_inset_center_bottom_2_by_3'), 
            aspect=(2, 3),
            props=set(['dark'])
        ),
        dict(
            obj=load_blend_model('./assets/walls.blend', 'wall_window_inset_center_top_2_by_3'), 
            aspect=(2, 3),
            props=set(['dark'])
        ),
        dict(
            obj=load_blend_model('./assets/walls.blend', 'wall_window_inset_top_2_by_3'), 
            aspect=(2, 3),
            props=set(['dark'])
        ),
    ],
    slab=[
        dict(
            obj=load_blend_model('./assets/steps.blend', 'slab')
        )
    ],
    steps_and_slab=[
        dict(
            obj=load_blend_model('./assets/steps.blend', 'steps_and_slab')
        )
    ],
    steps=[
        dict(
            obj=load_blend_model('./assets/steps.blend', 'steps_with_railing')
        ), 
        dict(
            obj=load_blend_model('./assets/steps.blend', 'steps_small')
        ), 
    ], 
    covering=[
        dict(
            obj=load_blend_model('./assets/roof_elements.blend', 'gabled')
        ), 
        dict(
            obj=load_blend_model('./assets/roof_elements.blend', 'conish')
        ), 
    ],
    pillar=[
        dict(
            obj=load_blend_hierarchy('./assets/pillars.blend', 'pillar_2')
        ), 
        dict(
            obj=load_blend_hierarchy('./assets/pillars.blend', 'pillar_3')
        ), 
        dict(
            obj=load_blend_hierarchy('./assets/pillars.blend', 'pillar_4')
        ), 
        dict(
            obj=load_blend_hierarchy('./assets/pillars.blend', 'pillar_5')
        ), 
        dict(
            obj=load_blend_hierarchy('./assets/pillars.blend', 'pillar_6')
        ), 
    ],
    balcony=[
        dict(
            obj=load_blend_hierarchy('./assets/balcony.blend', 'balcony')
        )
    ],
    balustrade=[
        dict(
            obj=load_blend_hierarchy('./assets/balcony.blend', 'balustrade_1')
        ),
        dict(
            obj=load_blend_hierarchy('./assets/balcony.blend', 'balustrade_2')
        ),
        dict(
            obj=load_blend_hierarchy('./assets/balcony.blend', 'balustrade_3')
        ),
        dict(
            obj=load_blend_hierarchy('./assets/balcony.blend', 'balustrade_4')
        ),
    ], 
    window=[
        dict(
            obj=load_blend_model('assets/window.blend', 'wind_low')
        )
    ],
    door=[
        dict(
            obj=load_blend_model('assets/door.blend', 'Mansion Door')
        )
    ],
    chimney=[
        dict(
            obj=load_blend_model('assets/chimney.blend', 'chimney')
        )
    ],
)
"""
