#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from itertools import combinations

class PositiveNegativeBuilder:
    def __init__(self, para_list:list, changes_list:list, authors:int) -> None:
        self.para_list = para_list
        self.changes_list = changes_list
        self.authors = authors
        self.groups_divide()

    def groups_divide(self):
        self.groups = []
        group_temp = []
        for id, para in enumerate(self.para_list):
            if id == 0:
                group_temp.append(para)
                continue
            if self.changes_list[id-1] == 1:
                self.groups.append(group_temp)
                group_temp = [para]
            elif self.changes_list[id-1] == 0:
                group_temp.append(para)
            if id == len(self.para_list) - 1:
                self.groups.append(group_temp)
        return self.groups

    def build_positive_instances(self):
        self.positive_instances = []
        for group in self.groups:
            if len(group) < 2:
                continue
            for comb in combinations(group,2):
                para_a, para_b = comb
                instances = (para_a, para_b, 0)
                self.positive_instances.append(instances)
        return self.positive_instances

    def build_negative_instances(self):
        self.negative_instances = []
        for id, group in enumerate(self.groups):
            if id == 0:
                group_pre = group
                continue
            for para_left in group_pre:
                for para_right in group:
                    instances = (para_left, para_right, 1)
                    self.negative_instances.append(instances)
            group_pre = group
        return self.negative_instances
    
    def build_negative_transparent(self):
        '''
        Situation for authors info was transparent
        '''
        assert sum(self.changes_list) + 1 == self.authors
        self.negative_instances = []
        for id, group in enumerate(self.groups):
            for para_a in group:
                rest_groups = self.groups.copy()
                rest_groups.pop(id)
                for r_group in rest_groups:
                    for para_b in r_group:
                        instances = (para_a, para_b, 1)
                        self.negative_instances.append(instances)
        return self.negative_instances

    def build_triple(self):
        '''
        This part (triple) was used for simcse, as below
        '''
        self.triple = []
        for id, group in enumerate(self.groups):
            if len(group) < 2:
                continue
            for comb in combinations(group,2):
                para_a, para_b = comb
                # if id != 0: # left
                #     for para_c in self.groups[id-1]:
                #         instances = (para_a, para_b, para_c)
                #         self.triple.append(instances)
                if id +1 != len(self.groups): # right
                    for para_c in self.groups[id+1]:
                        instances = (para_a, para_b, para_c)
                        self.triple.append(instances)
        return self.triple
    
    def build_triple_transparent(self):
        assert sum(self.changes_list) + 1 == self.authors
        self.triple = []
        for id, group in enumerate(self.groups):
            if len(group) < 2:
                continue
            for comb in combinations(group,2):
                para_a, para_b = comb
                rest_groups = self.groups.copy()
                rest_groups.pop(id)
                for r_group in rest_groups:
                    for para_c in r_group:
                        instances = (para_a, para_b, para_c)
                        self.triple.append(instances)
        return self.triple
    
    def build_all_negative(self):
        '''
        Situation in which all paragraphs are written by different authors
        '''
        assert sum(self.changes_list) == len(self.changes_list)
        self.all_negative = []
        for id, para in enumerate(self.para_list):
            if id == 0:
                para_pre = para
                continue
            instances = (para_pre, para)
            self.all_negative.append(instances)
            para_pre = para
        return self.all_negative


    def build_vaild_instances(self):
        '''
        This part is for vaild dataset of encoder
        '''
        self.vaild_instances = []
        for id, para in enumerate(self.para_list):
            if id == 0:
                para_pre = para
                continue
            instances = (para_pre, para, self.changes_list[id-1])
            self.vaild_instances.append(instances)
            para_pre = para
        return self.vaild_instances
        

if __name__ == "__main__":
    para_list = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"]
    changes_list = [0, 0, 1, 0, 0, 1, 1, 0, 1]
    my_dataset = PositiveNegativeBuilder(para_list, changes_list, 5)
    print(my_dataset.build_positive_instances())
    print(my_dataset.build_negative_instances())