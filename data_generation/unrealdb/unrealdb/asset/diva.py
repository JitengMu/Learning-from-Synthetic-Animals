class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d

MESH = ObjectView(dict())
MESH.WOMAN = "SkeletalMesh'/Game/T_pose_chara/rp_alison_rigged_001_yup_t.rp_alison_rigged_001_yup_t'"
MESH.MAN = "SkeletalMesh'/Game/T_pose_chara/rp_eric_rigged_001_yup_t.rp_eric_rigged_001_yup_t'"
MESH.GIRL1 = "SkeletalMesh'/Game/Girl_01/meshes/girl_01_j.girl_01_j'"

ANIM = ObjectView(dict())
ANIM.OpenExitClose2 = "AnimSequence'/Game/Mocap/skel_girl/girl_exit2_Char01.girl_exit2_Char01'"
ANIM.ClosingTrunk3 = "AnimSequence'/Game/Mocap/skel_girl/girl_CloseTrunk_Char00.girl_CloseTrunk_Char00'"
ANIM.OpenTrunk3 = "AnimSequence'/Game/Mocap/skel_girl/girl_OpenChunk_Char00.girl_OpenChunk_Char00'"
ANIM.OpenEnterClose2 = "AnimSequence'/Game/Mocap/skel_girl/girl_enter2_Char01.girl_enter2_Char01'"
ANIM.OpenExitClose1 = "AnimSequence'/Game/Animations/mocap_girl_0712_offsetnorm/openexitclose1_Char00.openexitclose1_Char00'"
ANIM.OpenExitClose4 = "AnimSequence'/Game/Animations/mocap_girl_0712_offsetnorm/openexitclose4_Char01.openexitclose4_Char01'"
ANIM.ClosingTrunk1 = "AnimSequence'/Game/Animations/mocap_girl_0712_offsetnorm/closingtrunk1_Char00.closingtrunk1_Char00'"
ANIM.ClosingTrunk2 = "AnimSequence'/Game/Animations/mocap_girl_0712_offsetnorm/closingtrunk2_Char00.closingtrunk2_Char00'"
ANIM.OpenTrunk1 = "AnimSequence'/Game/Animations/mocap_girl_0712_offsetnorm/opentrunk1_Char00.opentrunk1_Char00'"
ANIM.OpenTrunk2 = "AnimSequence'/Game/Animations/mocap_girl_0712_offsetnorm/opentrunk2_Char00.opentrunk2_Char00'"
ANIM.OpenEnterClose1 = "AnimSequence'/Game/Animations/mocap_girl_0712_offsetnorm/openenterclose1_Char02.openenterclose1_Char02'"
ANIM.OpenEnterClose3 = "AnimSequence'/Game/Animations/mocap_girl_0712_offsetnorm/openenterclose3_Char00.openenterclose3_Char00'"
ANIM.ManOpenExitClose1 = "AnimSequence'/Game/Animations/mocap_man_0712_offsetnorm/openexitclose1_Char01.openexitclose1_Char01'"
ANIM.ManOpenExitClose3 = "AnimSequence'/Game/Animations/mocap_man_0712_offsetnorm/openexitclose3_Char01.openexitclose3_Char01'"
ANIM.ManClosingTrunk1 = "AnimSequence'/Game/Animations/mocap_man_0712_offsetnorm/closingtrunk1_Char01.closingtrunk1_Char01'"
ANIM.ManClosingTrunk2 = "AnimSequence'/Game/Animations/mocap_man_0712_offsetnorm/closingtrunk2_Char01.closingtrunk2_Char01'"
ANIM.ManOpenTrunk1 = "AnimSequence'/Game/Animations/mocap_man_0712_offsetnorm/opentrunk1_Char01.opentrunk1_Char01'"
ANIM.ManOpenTrunk2 = "AnimSequence'/Game/Animations/mocap_man_0712_offsetnorm/opentrunk2_Char01.opentrunk2_Char01'"
ANIM.ManOpenEnterClose1 = "AnimSequence'/Game/Animations/mocap_man_0712_offsetnorm/openenterclose1_Char01.openenterclose1_Char01'"
ANIM.ManOpenEnterClose3 = "AnimSequence'/Game/Animations/mocap_man_0712_offsetnorm/openenterclose3_Char01.openenterclose3_Char01'"

anim_paths = [
    # human_mesh, anim_path, door_name, human_loc_rot, activity_type, if_flip
    (MESH.GIRL1, ANIM.OpenExitClose2,     'RF',    (-280,-340,125,0,200,0),  'OpenExitClose', False),
    (MESH.GIRL1, ANIM.OpenExitClose2,     'RR',    (-280,-460,125,0,200,0),  'OpenExitClose', False),
    (MESH.GIRL1, ANIM.OpenExitClose2,     'LF',    (-40,-330,125,0,200,0),  'OpenExitClose', True),
    (MESH.GIRL1, ANIM.OpenExitClose2,     'LR',    (-40,-430,125,0,200,0),  'OpenExitClose', True),
    (MESH.GIRL1, ANIM.ClosingTrunk3,      'Trunk', (-310,-490,125,0,160,0),   'Closing_Trunk', False),
    (MESH.GIRL1, ANIM.OpenTrunk3,         'Trunk', (-310,-490,125,0,160,0), 'Open_Trunk', False),
    (MESH.GIRL1, ANIM.OpenEnterClose2,    'RF',    (-370,-310,125,0,-200,0), 'OpenEnterClose', False),
    (MESH.GIRL1, ANIM.OpenEnterClose2,    'RR',    (-370,-420,125,0,-200,0), 'OpenEnterClose', False),
    (MESH.GIRL1, ANIM.OpenEnterClose2,    'LF',    (-130,-320,125,0,-200,0), 'OpenEnterClose', True),
    (MESH.GIRL1, ANIM.OpenEnterClose2,    'LR',    (-130,-430,125,0,-200,0), 'OpenEnterClose', True),
    (MESH.WOMAN, ANIM.OpenExitClose1,     'LF',    (-75,-170,125,0,180,0),   'Exiting', False),
    (MESH.WOMAN, ANIM.OpenExitClose1,     'LR',    (-75,-270,125,0,195,0),   'Exiting', False),
    (MESH.WOMAN, ANIM.OpenExitClose4,     'LF',    (-120,-160,125,0,180,0),  'Exiting', False),
    (MESH.WOMAN, ANIM.OpenExitClose4,     'LR',    (-150,-250,125,0,180,0),  'Exiting', False),
    (MESH.WOMAN, ANIM.OpenExitClose1,     'RF',    (-315,-170,125,0,180,0),   'Exiting', True),
    (MESH.WOMAN, ANIM.OpenExitClose1,     'RR',    (-315,-290,125,0,195,0),   'Exiting', True),
    (MESH.WOMAN, ANIM.OpenExitClose4,     'RF',    (-240,-160,125,0,180,0),  'Exiting', True),
    (MESH.WOMAN, ANIM.OpenExitClose4,     'RR',    (-280,-250,125,0,180,0),  'Exiting', True),
    (MESH.WOMAN, ANIM.ClosingTrunk1,      'Trunk', (-230,-470,125,0,180,0),  'Closing_Trunk', False),
    (MESH.WOMAN, ANIM.ClosingTrunk2,      'Trunk', (-160,-490,125,0,-20,0),   'Closing_Trunk', False),
    (MESH.WOMAN, ANIM.OpenTrunk1,         'Trunk', (-230,-480,125,0,180,0), 'Open_Trunk', False),
    (MESH.WOMAN, ANIM.OpenTrunk2,         'Trunk', (-210,-490,125,0,-20,0),   'Open_Trunk', False),
    (MESH.WOMAN, ANIM.OpenEnterClose1,    'LR',    (-80,-315,125,0,180,0),  'Entering', False),
    (MESH.WOMAN, ANIM.OpenEnterClose1,    'LF',    (-80,-210,125,0,180,0),  'Entering', False),
    (MESH.WOMAN, ANIM.OpenEnterClose3,    'LR',    (-95,-295,125,0,-110,0), 'Entering', False),
    (MESH.WOMAN, ANIM.OpenEnterClose3,    'LF',    (-90,-195,125,0,-110,0), 'Entering', False),
    (MESH.WOMAN, ANIM.OpenEnterClose1,    'RR',    (-330,-315,125,0,180,0),  'Entering', True),
    (MESH.WOMAN, ANIM.OpenEnterClose1,    'RF',    (-330,-210,125,0,180,0),  'Entering', True),
    (MESH.WOMAN, ANIM.OpenEnterClose3,    'RR',    (-305,-275,125,0,-90,0), 'Entering', True),
    (MESH.WOMAN, ANIM.OpenEnterClose3,    'RF',    (-300,-175,125,0,-90,0), 'Entering', True),
    (MESH.MAN,   ANIM.ManOpenExitClose1,  'LF',    (-75,-170,125,0,180,0),  'Exiting', False),
    (MESH.MAN,   ANIM.ManOpenExitClose1,  'LR',    (-75,-280,125,0,195,0),  'Exiting', False),
    (MESH.MAN,   ANIM.ManOpenExitClose3,  'LF',    (-100,-160,125,0,-90,0), 'Exiting', False),
    (MESH.MAN,   ANIM.ManOpenExitClose3,  'LR',    (-100,-265,125,0,-90,0), 'Exiting', False),
    (MESH.MAN,   ANIM.ManOpenExitClose1,  'RF',    (-305,-170,125,0,180,0),  'Exiting', True),
    (MESH.MAN,   ANIM.ManOpenExitClose1,  'RR',    (-305,-280,125,0,195,0),  'Exiting', True),
    (MESH.MAN,   ANIM.ManOpenExitClose3,  'RF',    (-290,-160,125,0,-90,0), 'Exiting', True),
    (MESH.MAN,   ANIM.ManOpenExitClose3,  'RR',    (-290,-265,125,0,-90,0), 'Exiting', True),
    (MESH.MAN,   ANIM.ManClosingTrunk1,   'Trunk', (-230,-470,125,0,180,0), 'Closing_Trunk', False),
    (MESH.MAN,   ANIM.ManClosingTrunk2,   'Trunk', (-200,-490,125,0,-30,0),    'Closing_Trunk', False),
    (MESH.MAN,   ANIM.ManOpenTrunk1,      'Trunk', (-230,-475,125,0,180,0), 'Open_Trunk', False),
    (MESH.MAN,   ANIM.ManOpenTrunk2,      'Trunk', (-200,-480,125,0,-30,0),   'Open_Trunk', False),
    (MESH.MAN,   ANIM.ManOpenEnterClose1, 'LR',    (-70,-325,125,0,180,0),  'Entering', False),
    (MESH.MAN,   ANIM.ManOpenEnterClose1, 'LF',    (-70,-210,125,0,180,0),  'Entering', False),
    (MESH.MAN,   ANIM.ManOpenEnterClose3, 'LR',    (-100,-300,125,0,-110,0), 'Entering', False),
    (MESH.MAN,   ANIM.ManOpenEnterClose3, 'LF',    (-100,-210,125,0,-110,0), 'Entering', False),
    (MESH.MAN,   ANIM.ManOpenEnterClose1, 'RR',    (-310,-325,125,0,180,0),  'Entering', True),
    (MESH.MAN,   ANIM.ManOpenEnterClose1, 'RF',    (-310,-210,125,0,180,0),  'Entering', True),
    (MESH.MAN,   ANIM.ManOpenEnterClose3, 'RR',    (-320,-260,125,0,-80,0), 'Entering', True),
    (MESH.MAN,   ANIM.ManOpenEnterClose3, 'RF',    (-320,-170,125,0,-80,0), 'Entering', True),
]