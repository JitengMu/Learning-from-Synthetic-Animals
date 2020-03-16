
# class ObjectView(object):
#     def __init__(self, d):
#         self.__dict__ = d
#     # Write some test cases for this ObjectView

class ObjectView(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

animal_mesh = ObjectView({
    "african_elephant": "SkeletalMesh'/Game/Animal_pack_ultra_2/Meshes/African_elephant/SK_African_elephant.SK_African_elephant'",
    "bat": "SkeletalMesh'/Game/Animal_pack_ultra_2/Meshes/Bat/SK_bat.SK_bat'",
    "beagle": "SkeletalMesh'/Game/Animal_pack_ultra_2/Meshes/Beagle/SK_beagle.SK_beagle'",
    "box_turtle": "SkeletalMesh'/Game/Animal_pack_ultra_2/Meshes/Box_turtle/SK_box_turtle.SK_box_turtle'",
    "camel": "SkeletalMesh'/Game/Animal_pack_ultra_2/Meshes/Camel/SK_camel.SK_camel'",
    "cane_corso": "SkeletalMesh'/Game/Animal_pack_ultra_2/Meshes/Cane_corso/SK_cane_corso.SK_cane_corso'",
    "cape_buffalo": "SkeletalMesh'/Game/Animal_pack_ultra_2/Meshes/Cape_buffalo/SK_Cape_Buffalo.SK_Cape_Buffalo'",
    "cat": "SkeletalMesh'/Game/Animal_pack_ultra_2/Meshes/Cat_v1/SK_cat.SK_cat'",
    "celtic_wolfhound": "SkeletalMesh'/Game/Animal_pack_ultra_2/Meshes/Celtic_wolfhound/SK_celtic_wolfhound.SK_celtic_wolfhound'",
    "chick": "SkeletalMesh'/Game/Animal_pack_ultra_2/Meshes/Chick/SK_chick.SK_chick'",
    "comodo_dragon": "SkeletalMesh'/Game/Animal_pack_ultra_2/Meshes/Comodo_dragon/SK_comodo_dragon.SK_comodo_dragon'",
    "domestic_pig": "SkeletalMesh'/Game/Animal_pack_ultra_2/Meshes/Domestic_pig/SK_domestic_pig.SK_domestic_pig'",
    "domestic_sheep": "SkeletalMesh'/Game/Animal_pack_ultra_2/Meshes/Domestic_sheep/SK_Domestic_sheep.SK_Domestic_sheep'",
    "goliath_spider": "SkeletalMesh'/Game/Animal_pack_ultra_2/Meshes/Goliath_spider/SK_goliath_spider.SK_goliath_spider'",
    "green_lizard": "SkeletalMesh'/Game/Animal_pack_ultra_2/Meshes/Green_lizard_v1/SK_green_lizard.SK_green_lizard'",
    "hellenic_hound": "SkeletalMesh'/Game/Animal_pack_ultra_2/Meshes/Helenic_hound/SK_hellenic_hound.SK_hellenic_hound'",
    "horse": "SkeletalMesh'/Game/Animal_pack_ultra_2/Meshes/Horse/SK_horse_skeleton.SK_horse_skeleton'",
    "indian_elephant": "SkeletalMesh'/Game/Animal_pack_ultra_2/Meshes/Indian_elephant/SK_Indian_elephant.SK_Indian_elephant'",
    "leopard": "SkeletalMesh'/Game/Animal_pack_ultra_2/Meshes/Leopard/SK_leopard.SK_leopard'",
    "longhorn_cattle": "SkeletalMesh'/Game/Animal_pack_ultra_2/Meshes/Longhorn_cattle/SK_Longhorn_cattle.SK_Longhorn_cattle'",
    "longhorn_cattle_v2": "SkeletalMesh'/Game/Animal_pack_ultra_2/Meshes/Longhorn_cattle_v2/SK_Longhorn_cattle_V2.SK_Longhorn_cattle_V2'",
    "mud_pig": "SkeletalMesh'/Game/Animal_pack_ultra_2/Meshes/Mud_pig/SK_mud_pig.SK_mud_pig'",
    "penguin": "SkeletalMesh'/Game/Animal_pack_ultra_2/Meshes/Penguin/SK_penguin.SK_penguin'",
    "pug": "SkeletalMesh'/Game/Animal_pack_ultra_2/Meshes/Pug/SK_pug.SK_pug'",
    "rhino": "SkeletalMesh'/Game/Animal_pack_ultra_2/Meshes/Rhino/SK_rhino.SK_rhino'",
    "scotland_cattle": "SkeletalMesh'/Game/Animal_pack_ultra_2/Meshes/Scotland_cattle/SK_Scotland_cattle.SK_Scotland_cattle'",
    "snapping_turtle": "SkeletalMesh'/Game/Animal_pack_ultra_2/Meshes/Snapping_turtle/SK_snapping_turtle.SK_snapping_turtle'",
    "tiger": "SkeletalMesh'/Game/Animal_pack_ultra_2/Meshes/Tiger/SK_tiger.SK_tiger'",
    "tucano_bird": "SkeletalMesh'/Game/Animal_pack_ultra_2/Meshes/Tucano_bird_v1/SK_tucano_bird.SK_tucano_bird'",
    "zebra": "SkeletalMesh'/Game/Animal_pack_ultra_2/Meshes/Zebra/SK_zebra.SK_zebra'",
})

horse_material = ObjectView({
    "v1": "Material'/Game/Animal_pack_ultra_2/Materials/M_Horse_material.M_Horse_material'",
    "v2": "Material'/Game/Animal_pack_ultra_2/Materials/M_Horse_v2_material.M_Horse_v2_material'",
    "v3": "Material'/Game/Animal_pack_ultra_2/Materials/M_Horse_v3_material.M_Horse_v3_material'",
    "v4": "Material'/Game/Animal_pack_ultra_2/Materials/M_Horse_v4_material.M_Horse_v4_material'",
    "v5": "Material'/Game/Animal_pack_ultra_2/Materials/M_Horse_v5_material.M_Horse_v5_material'",
    "v6": "Material'/Game/Animal_pack_ultra_2/Materials/M_Horse_v6_material.M_Horse_v6_material'",
})