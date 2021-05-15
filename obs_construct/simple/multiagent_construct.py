import numpy as np
import copy

class ObsConstruct:
    def __init__(self, size_x, size_y, detector_num, fighter_num):
        self.battlefield_size_x = size_x
        self.battlefield_size_y = size_y
        self.detector_num = detector_num
        self.fighter_num = fighter_num
        self.num_features = 10      # 要使用的特征数目
        self.img_obs_reduce_ratio = 10  # 地图的缩放比例
        self.embedding_size = None

    def obs_construct(self, obs_raw_dict):
        obs_dict = {} 

        detector_data_obs_list = obs_raw_dict['detector_obs_list']
        fighter_data_obs_list = obs_raw_dict['fighter_obs_list']
        joint_data_obs_dict = obs_raw_dict['joint_obs_dict']

        detector_embedding, fighter_embedding = self.__get_obs(detector_data_obs_list, fighter_data_obs_list, joint_data_obs_dict)
        enemy_embedding = self.__get_enemyunit_obs(detector_data_obs_list, fighter_data_obs_list, joint_data_obs_dict)
        alive_status = self.__get_alive_status(detector_data_obs_list, fighter_data_obs_list)

        obs_dict['detector'] = detector_embedding
        obs_dict['fighter'] = fighter_embedding
        obs_dict['enemy'] = enemy_embedding
        obs_dict['alive'] = alive_status

        return obs_dict


    def __get_obs(self, detector_data_obs_list, fighter_data_obs_list, joint_data_obs_dict):
        detector_embedding_outer = []
        fighter_embedding_outer = []

        for det_num in range(self.detector_num):
            if detector_data_obs_list[det_num]['alive']:
                detector_embedding_inner = []
                detector_embedding_inner.extend(self._feature_embed(
                    detector_data_obs_list[det_num]['id'], embed_type='plain'))
                detector_embedding_inner.extend(self._feature_embed(
                    detector_data_obs_list[det_num]['alive'], embed_type='plain'))
                detector_embedding_inner.extend(self._feature_embed(
                    detector_data_obs_list[det_num]['pos_x'], embed_type='plain'))
                detector_embedding_inner.extend(self._feature_embed(
                    detector_data_obs_list[det_num]['pos_y'], embed_type='plain'))
                detector_embedding_inner.extend(self._feature_embed(
                    detector_data_obs_list[det_num]['course'], embed_type='plain'))
                detector_embedding_inner.extend(self._feature_embed(
                    detector_data_obs_list[det_num]['r_iswork'], embed_type='plain'))
                detector_embedding_inner.extend(self._feature_embed(
                    detector_data_obs_list[det_num]['r_fre_point'], embed_type='plain'))
            else:
                detector_embedding_inner = [0.]

            detector_embedding_outer.append(detector_embedding_inner)
        
        # 由于攻击单元的数目是在探测单元的基础上计数的，所以需要加上探测单元数目
        for fig_num in range(self.fighter_num):
            if fighter_data_obs_list[fig_num]['alive']:
                fighter_embedding_inner = []
                fighter_embedding_inner.extend(self._feature_embed(
                    fighter_data_obs_list[fig_num]['id'], embed_type='plain'))
                fighter_embedding_inner.extend(self._feature_embed(
                    fighter_data_obs_list[fig_num]['alive'], embed_type='plain'))
                fighter_embedding_inner.extend(self._feature_embed(
                    fighter_data_obs_list[fig_num]['pos_x'], embed_type='plain'))
                fighter_embedding_inner.extend(self._feature_embed(
                    fighter_data_obs_list[fig_num]['pos_y'], embed_type='plain'))
                fighter_embedding_inner.extend(self._feature_embed(
                    fighter_data_obs_list[fig_num]['course'], embed_type='plain'))
                fighter_embedding_inner.extend(self._feature_embed(
                    fighter_data_obs_list[fig_num]['r_iswork'], embed_type='plain'))
                fighter_embedding_inner.extend(self._feature_embed(
                    fighter_data_obs_list[fig_num]['r_fre_point'], embed_type='plain'))
                # fighter_embedding_inner.extend(self._feature_embed(
                #     fighter_data_obs_list[fig_num]['l_missle_left'], embed_type='plain'))
                # fighter_embedding_inner.extend(self._feature_embed(
                #     fighter_data_obs_list[fig_num]['s_missle_left'], embed_type='plain'))
                if self.embedding_size is None:
                    self.embedding_size = len(fighter_embedding_inner)
            else:
                fighter_embedding_inner = [0.]


            fighter_embedding_outer.append(fighter_embedding_inner)

        detector_embedding_outer = self._align_rowvector(detector_embedding_outer)
        fighter_embedding_outer = self._align_rowvector(fighter_embedding_outer)

        return detector_embedding_outer, fighter_embedding_outer


    def __get_enemyunit_obs(self, detector_data_obs_list, fighter_data_obs_list, joint_data_obs_dict):
        # 目前考虑的单位只有 fighter，而且在目前的设定中，敌方单位数量和我方单位数量相等，这样的条件下这里的循环才不存在逻辑问题。
        enemy_units = {i + 1: None for i in range(self.fighter_num)}
        enemy_unit_embedding_outer = []

        for num in range(self.detector_num):
            for unit in detector_data_obs_list[num]['r_visible_list']:
                if enemy_units[unit['id']] is None:
                    enemy_units[unit['id']] = unit

        for num in range(self.fighter_num):
            for unit in fighter_data_obs_list[num]['r_visible_list']:
                if enemy_units[unit['id']] is None:
                    enemy_units[unit['id']] = unit

        for num in range(self.fighter_num):
            if enemy_units[num + 1] is not None:
                unit = enemy_units[num + 1]
                enemy_unit_embedding_inner = []
                enemy_unit_embedding_inner.extend(self._feature_embed(unit['id'], embed_type='plain'))
                enemy_unit_embedding_inner.extend(self._feature_embed(unit['type'], embed_type='plain'))
                enemy_unit_embedding_inner.extend(self._feature_embed(unit['pos_x'], embed_type='plain'))
                enemy_unit_embedding_inner.extend(self._feature_embed(unit['pos_y'], embed_type='plain'))
            else:
                enemy_unit_embedding_inner = [0.]
            enemy_unit_embedding_outer.append(enemy_unit_embedding_inner)
        
        enemy_unit_embedding_outer = self._align_rowvector(enemy_unit_embedding_outer)
        return np.asarray(enemy_unit_embedding_outer)


    def __get_alive_status(self, detector_data_obs_list, fighter_data_obs_list):
        alive_status = np.full((self.detector_num + self.fighter_num, 1), True)
        for x in range(self.detector_num):
            if not detector_data_obs_list[x]['alive']:
                alive_status[x][0] = False
        for x in range(self.fighter_num):
            if not fighter_data_obs_list[x]['alive']:
                alive_status[x + self.detector_num][0] = False
        return alive_status


    def _align_rowvector(self, nested_list):
        # 对齐两层嵌套列表 nested_list，并将其转换为 np.array 类型返回
        for inner_list in nested_list:
            if len(inner_list) < self.embedding_size:
                inner_list.extend([0.] * (self.embedding_size - len(inner_list)))

        return np.asarray(nested_list)


    def _feature_embed(self, feature, embed_type):
        """
        参数：
        feature 是一个数，对其进行编码
        onehot：独热编码，除了要知道feature的值，还需要给出最大长度
        normalize：归一化编码，需要给出归一化区间的上下界是多少
        plain：不作编码
        返回值为一个列表
        """
        assert embed_type in ('onehot', 'normalize', 'plain')
        if embed_type == 'plain':
            return list([feature])
        if embed_type == 'normalize':
            pass
        if embed_type == 'onehot':
            pass

    