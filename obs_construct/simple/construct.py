import numpy as np
import copy
import math

class ObsConstruct:
    def __init__(self, size_x, size_y, detector_num, fighter_num):
        self.battlefield_size_x = size_x
        self.battlefield_size_y = size_y
        self.detector_num = detector_num    # 其实也是地我方单位的最大数目
        self.fighter_num = fighter_num
        self.num_features = 10      # 要使用的特征数目
        self.img_obs_reduce_ratio = 10  # 地图的缩放比例
        self.max_embedding_size = None

    def obs_construct(self, obs_raw_dict):
        obs_dict = {}
        detector_list = []
        fighter_list = []

        detector_data_obs_list = obs_raw_dict['detector_obs_list']
        fighter_data_obs_list = obs_raw_dict['fighter_obs_list']
        joint_data_obs_dict = obs_raw_dict['joint_obs_dict']
#         for unit in fighter_data_obs_list:
#             print(unit['id'], unit['alive'])
        alive_status = self.__get_alive_status(detector_data_obs_list, fighter_data_obs_list)
        detector_embedding, fighter_embedding = self.__get_obs(detector_data_obs_list, fighter_data_obs_list, joint_data_obs_dict)
        enemy_embedding = self.__get_enemyunit_obs(detector_data_obs_list, fighter_data_obs_list, joint_data_obs_dict)
        
        fighter_visible_enemys_dict = self.__get_fighter_visible_enemys(fighter_data_obs_list)
        
        # 将 detector_embedding, fighter_embedding 的第三个维度标记为数目
        # detector_embedding = detector_embedding[None, :, :]
        # detector_embedding = detector_embedding.repeat(self.detector_num, axis=0)
        # fighter_embedding = fighter_embedding[None, :, :]
        # fighter_embedding = fighter_embedding.repeat(self.fighter_num, axis=0)
        # print('np.array(fighter_embedding).shape',np.array(fighter_embedding).shape)
        # for num in range(self.fighter_num):
            
        #     fighter_embedding[num, num, :] = -np.ones(fighter_embedding.shape[2])
        #     tmp_slice = copy.deepcopy(fighter_embedding[num, :, :])
        #     fighter_list.append({'obs': tmp_slice, 'alive': alive_status[num + self.detector_num][0]})

        # obs_dict['detector'] = detector_embedding

        obs_dict['fighter'] = fighter_embedding
        obs_dict['enemy'] = enemy_embedding
        obs_dict['alive'] = alive_status[:, 0]
        obs_dict['fighter_visible_enemys_dict'] = fighter_visible_enemys_dict
        obs_dict['fighter_raw'] = fighter_data_obs_list

        return obs_dict

    def __get_fighter_visible_enemys(self, fighter_data_obs_list):
        fighter_visible_enemys_dict = {}
        for num in range(self.fighter_num):
            enemy_units_ids = []
            for unit in fighter_data_obs_list[num]['r_visible_list']:
                enemy_units_ids.append(unit['id'])
            if len(enemy_units_ids) > 0:
                fighter_visible_enemys_dict[fighter_data_obs_list[num]['id']] = enemy_units_ids
           
        return fighter_visible_enemys_dict

    def __get_alive_status(self, detector_data_obs_list, fighter_data_obs_list):
        alive_status = np.full((self.detector_num+self.fighter_num, 1),True)
        for x in range(self.detector_num):
            if not detector_data_obs_list[x]['alive']:
                alive_status[x][0] = False
        for x in range(self.fighter_num):
            if not fighter_data_obs_list[x]['alive']:
                alive_status[x+self.detector_num][0] = False
        return alive_status


    def __get_obs(self, detector_data_obs_list, fighter_data_obs_list, joint_data_obs_dict):
        # 返回我方单位的嵌入，一个二维的 np.array
        detector_embedding_outer = []
        fighter_embedding_outer = []
        id_list = []
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
            id_list.append((fighter_data_obs_list[fig_num]['id'], fighter_data_obs_list[fig_num]['alive']))
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
            else:
                fighter_embedding_inner = [0.]

            fighter_embedding_outer.append(fighter_embedding_inner)
        
        # 此处得到嵌入维度，由于底层平台给出的我方单位的信息就比敌方单位多，因此嵌入维度要比敌方维度长，此处得到我方单位的嵌入维度
        # 之后在做敌方单位的特征嵌入的时候，需要保持和我方单位一样长
        if self.max_embedding_size is None:
            self.max_embedding_size = len(fighter_embedding_outer[0])
        detector_embedding_outer = self._align_rowvector(detector_embedding_outer,self.max_embedding_size)
        fighter_embedding_outer = self._align_rowvector(fighter_embedding_outer,self.max_embedding_size)        
        return detector_embedding_outer, fighter_embedding_outer


    def __get_enemyunit_obs(self, detector_data_obs_list, fighter_data_obs_list, joint_data_obs_dict):
        # 返回敌方单位的嵌入，来自于所有我方单位的['r_visible_list']，是一个二维的 np.array 
        enemy_units = []
        exist_unit_id = set()
        enemy_unit_embedding_outer = [[] for i in range(10) ]


        for num in range(self.detector_num):
            for unit in detector_data_obs_list[num]['r_visible_list']:
                enemy_units.append(unit)
        for num in range(self.fighter_num):
            for unit in fighter_data_obs_list[num]['r_visible_list']:
                enemy_units.append(unit)

        for unit in enemy_units:
#             print(exist_unit_id)
            if unit['id'] not in exist_unit_id:
                exist_unit_id.add(unit['id'])
                enemy_unit_embedding_inner = []
                enemy_unit_embedding_inner.extend(self._feature_embed(unit['id'], embed_type='plain'))
                enemy_unit_embedding_inner.extend(self._feature_embed(unit['type'], embed_type='plain'))
                enemy_unit_embedding_inner.extend(self._feature_embed(unit['pos_x'], embed_type='plain'))
                enemy_unit_embedding_inner.extend(self._feature_embed(unit['pos_y'], embed_type='plain'))
                embedding_size = len(enemy_unit_embedding_inner)

                # 特征维度的对齐
                for _ in range(self.max_embedding_size - embedding_size):
                    enemy_unit_embedding_inner.append(0.) 
                enemy_unit_embedding_outer[int(unit['id'])-1].extend(enemy_unit_embedding_inner)

        for i in range(len(enemy_unit_embedding_outer)):
            if len(enemy_unit_embedding_outer[i]) == 0:
                enemy_unit_embedding_outer[i] = [0] * self.max_embedding_size

        # 单位长度的对齐
        for _ in range(self.detector_num + self.fighter_num - len(enemy_unit_embedding_outer)):
            enemy_unit_embedding_outer.append([0.])
        enemy_unit_embedding_outer = self._align_rowvector(enemy_unit_embedding_outer, length=self.max_embedding_size)

        return enemy_unit_embedding_outer


    def _align_rowvector(self, nested_list, length=None):
        # 对齐两层嵌套列表 nested_list，并将其转换为矩阵返回；
        # 参数length为对其长度：如果给出了该参数，则所有行向量对齐到length长度；否则对齐到最长的长度
        inner_length = -1 if length is None else length

        for inner_list in nested_list:
            if len(inner_list) > inner_length:
                inner_length = len(inner_list)

        for inner_list in nested_list:
            current_len = len(inner_list)
            inner_list.extend([0.] * (inner_length - current_len))

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

  
