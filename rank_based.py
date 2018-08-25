#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description: 

import sys
import math
import random
import numpy as np

from  baselines.her.binary_heap import BinaryHeap


"""
Important point
self.learn_start > self.batch_size
and self.learn_start > size of partition, i.e., self.size/partition_num
This is because we need to take the first partition at least, as there
is a floor operation in sample
"""


class Experience(object):

    def __init__(self, conf):
        self.size = conf['size']
        # If the transitions should be replaced if the heap is full
        # A lower priority node is expelled
        self.replace_flag = conf['replace_old'] if 'replace_old' in conf else True
        # Represents the max capacity of the heap
        self.priority_size = conf['priority_size'] if 'priority_size' in conf else self.size

        # The alpha used in Equation (1) in PER paper
        self.alpha = conf['alpha'] if 'alpha' in conf else 0.7
        # The bias correction term. Usually annealed linearly from 
        # beta_zero to 1. Section 3.4 of the paper
        self.beta_zero = conf['beta_zero'] if 'beta_zero' in conf else 0.5
        self.batch_size = conf['batch_size'] if 'batch_size' in conf else 32
        self.learn_start = conf['learn_start'] if 'learn_start' in conf else 32
        self.total_steps = conf['steps'] if 'steps' in conf else 100000
        # partition number N, split total size to N part
        self.partition_num = conf['partition_num'] if 'partition_num' in conf else 100
        self.partition_size = conf['partition_size'] if 'partition_size' in conf else math.floor(self.size / self.partition_num)
        if 'partition_size' in conf:
            self.partition_num = math.floor(self.size / self.partition_size)

        self.index = 0
        self.record_size = 0
        self.isFull = False

        self._experience = {}
        self.priority_queue = BinaryHeap(self.priority_size)

        # Added in new code
        self.distribution = None
        self.dist_index = 1

        self.beta_grad = (1 - self.beta_zero) / (self.total_steps - self.learn_start)

        # Debug Code
        self.debug = {}


    # Return the correct distribution, build if required
    def return_distribution(self, dist_index):
        if (dist_index == self.dist_index) and self.dist_index > 1:
            return self.distribution
        elif dist_index < self.dist_index:
            # print("Dist_index is: "+str(dist_index))
            # print("Self.dist_index is: "+str(self.dist_index))
            raise Exception('Elements have been illegally deleted from the priority_queue in rank_based')
        else:
            res = {}
            # Store the current dist_index
            self.dist_index = dist_index
            partition_num = dist_index

            # The procedure being followed here is that given on Page 13
            # last line. We divide the whole range into 'k' segments
            # This has the advantage that the same transition will not be
            # picked twice in the same batch (Stratified Sampling)
            n = partition_num * self.partition_size

            if self.batch_size <= n <= self.priority_size:
                distribution = {}
                # P(i) = (rank i) ^ (-alpha) / sum ((rank i) ^ (-alpha))
                pdf = list(
                    map(lambda x: math.pow(x, -self.alpha), range(1, n + 1))
                )
                pdf_sum = math.fsum(pdf)
                distribution['pdf'] = list(map(lambda x: x / pdf_sum, pdf))
                # split to k segment, and than uniform sample in each k
                # set k = batch_size, each segment has total probability is 1 / batch_size
                # strata_ends keep each segment start pos and end pos

                # The following code creates strata_ends such that 
                # strata_ends[i]-strata_ends[0] = (i-1)*1/batch_size probability
                cdf = np.cumsum(distribution['pdf'])
                strata_ends = {1: 0, self.batch_size + 1: n}
                step = 1 / self.batch_size
                index = 1
                for s in range(2, self.batch_size + 1):
                    while cdf[index] < step:
                        index += 1
                    strata_ends[s] = index
                    step += 1 / self.batch_size

                distribution['strata_ends'] = strata_ends

                # print("The strata is: "+str(distribution['strata_ends']))

                self.distribution = distribution
        return self.distribution




    def fix_index(self):
        """
        get next insert index
        :return: index, int
        """
        if self.record_size <= self.size:
            # self.record_size increases till self.size and stays there after that
            self.record_size += 1
        
        # self.index is being monotonically increased and thus say self.size = 3
        # When self.index = 3, the heap is full and when self.index = 6, three
        # other elements have been added and hence it is still full
        # This will happen because replace is True

        # But self.index is being set to 1 when replace_flag is true. Hence I don't
        # see why % operator was used
        if self.index % self.size == 0:
            # This condition because self.index = 0 initially, so control will always reach here
            self.isFull = True if len(self._experience) == self.size else False
            if self.replace_flag:
                self.index = 1
                # Doubt::  Won't the highest priority node be replaced?
                return self.index
            else:
                sys.stderr.write('Experience replay buff is full and replace is set to FALSE!\n')
                return -1
        else:
            self.index += 1
            return self.index

    def store(self, experience):
        """
        store experience, suggest that experience is a tuple of (s1, a, r, s2, g, t)
        so each experience is valid
        :param experience: maybe a tuple, or list
        :return: bool, indicate insert status
        """

        # This function should ideally be called only if a new experience needs to 
        # be stored because it is given the highest priority

        # Get the next position to be inserted in
        insert_index = self.fix_index()
        if insert_index > 0:
            # Remove the previous experience with the same index
            if insert_index in self._experience:
                del self._experience[insert_index]
            # Add the newest experience
            self._experience[insert_index] = experience

            ######Debug
            self._experience[insert_index]['new'] = True
            ######

            # add to priority queue
            # Add it with max priority so that it gets picked as soon as possible
            priority = self.priority_queue.get_max_priority()
            # Update the node where the new experience was inserted
            self.priority_queue.update(priority, insert_index)
            # print('The buffer size is: '+str(self.record_size))
            return True
        else:
            # This happens if replace is set to false and elements
            # are trying to be added
            sys.stderr.write('Insert failed\n')
            return False

    def retrieve(self, indices):
        """
        get experience from indices
        :param indices: list of experience id
        :return: experience replay sample
        """
        
        # self.debug['old'] = 0
        # self.debug['new'] = 0

        # #######Debug
        # for v in indices:
        #     if 'new' in self._experience[v].keys():
        #         self.debug['new'] += 1
        #         del self._experience[v]['new']
        #     else:
        #         self.debug['old'] += 1

        # f = open('new_old_ratio.txt', 'a')
        # f.write("The ratio is: "+str(float(self.debug['new'])/self.debug['old'])+'\n')

        # #######

        # Given a list of Experience_IDs, return the experience
        # it represents

        return [self._experience[v] for v in indices]

    def rebalance(self):
        """
        rebalance priority queue
        :return: None
        """

        # Balance the Heap from scratch
        self.priority_queue.balance_tree()

    # Function called from rank_based_test.py
    def update_priority(self, indices, delta):
        """
        update priority according indices and deltas
        :param indices: list of experience id
        :param delta: list of delta, order correspond to indices
        :return: None
        """

        # Update the priority of the node. indices[i] should represent
        # the experience_ID and delta should represent TD error (priority) --- Check this

        # Update the priorities of multiple nodes, given their experience_ids
        # and new priorities
        for i in range(0, len(indices)):
            self.priority_queue.update(math.fabs(delta[i]), indices[i])

    # if batch_size argument is passed, use that, else use the one at __init__
    def sample(self, global_step, uniform_priority=False, batch_size=32):
        """
        sample a mini batch from experience replay
        :param global_step: now training step
        :return: experience, list, samples
        :return: w, list, weights
        :return: rank_e_id, list, samples id, used for update priority
        """
        if self.record_size < self.learn_start:
            # Store a minimum of self.learn_start number of experiences before starting
            # any kind of learning. This is done to ensure there is not learning happening
            # with very small number of examples, leading to unstable estimates
            # Recollect: self.record_size increases till it reaches self.size and then stops there
            sys.stderr.write('Record size less than learn start! Sample failed\n')
            return False, False, False

        # If the replay buffer is not full, find the right partition to use
        # If only half the buffer is full, partition number 'self.partition_num/2'
        # is used because there are only those many ranks assigned

        # dist_index will always be the last partition after the replay
        # buffer is full. If it is not full, it will represent some
        # partition number less than that
        # print("(In rank_based_new.py) Values are (record_size, size, partition_num)::"+str(self.record_size)+"::"+str(self.size)+"::"+str(self.partition_num))
        dist_index = math.floor(self.record_size / self.size * self.partition_num)
        # dist_index = max(math.floor(self.record_size / self.size * self.partition_num)+1, self.partition_num)
        # issue 1 by @camigord
        partition_size = math.floor(self.size / self.partition_num)
        partition_max = dist_index * partition_size
        
        ############################
        # distribution = self.distributions[dist_index]
        # print("Dist Index is: "+str(dist_index))
        distribution = self.return_distribution(dist_index)
        ############################

        # print("Dist Index is: "+str(dist_index))

        rank_list = []
        # sample from k segments


        if uniform_priority==True:
            for i in range(1, self.batch_size + 1):
                index = random.randint(1,distribution['strata_ends'][self.batch_size]+1)
                rank_list.append(index)
            # The following statement is to ensure the there is no bias correction when uniform sampliing is done
            w = np.ones(self.batch_size)

        else:
            # This is stratified sampling. Each segment represents a probability
            # of 1/self.batch_size and we sample once from each segment
            # index represents which rank to choose, (1,n)
            for n in range(1, self.batch_size + 1):
                if distribution['strata_ends'][n] + 1 >= distribution['strata_ends'][n + 1]:
                    index = distribution['strata_ends'][n + 1]
                else:
                    # print("The values are: "+str(distribution['strata_ends'][n] + 1)+"::"+str(distribution['strata_ends'][n + 1]))
                    index = random.randint(distribution['strata_ends'][n] + 1,
                                           distribution['strata_ends'][n + 1])
                rank_list.append(index)

            # beta, increase by global_step (the current training step), max 1
            # This is linear annealing mentioned in section 3.4
            beta = min(self.beta_zero + (global_step - self.learn_start - 1) * self.beta_grad, 1)
            # find all alpha pow, notice that pdf is a list, start from 0
            alpha_pow = [distribution['pdf'][v - 1] for v in rank_list]
            # w = (N * P(i)) ^ (-beta) / max w
            # This equation is metioned in equation 3.4
            w = np.power(np.array(alpha_pow) * partition_max, -beta)
            w_max = max(w)
            w = np.divide(w, w_max)
            # rank list is priority id
            # convert to experience id

        # This gives all the experience_IDs based on the priority (node numbers)
        rank_e_id = self.priority_queue.priority_to_experience(rank_list)
        # get experience id according rank_e_id
        experience = self.retrieve(rank_e_id)
        return experience, w, rank_e_id
 