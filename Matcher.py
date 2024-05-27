import torch
import numpy as np
ERROR = 10
class Matcher():

    def __init__(self, threshold=0.5, buffer_size=1, lambda_value=0.8):

        """
        Initialize the Matcher class with threshold, buffer size, and lambda value.

        Args:
        - threshold (float): Threshold for matching objects
        - buffer_size (int): Size limit of the object buffer
        - lambda_value (float): Lambda value for re-ranking
        """

        self.threshold = threshold
        self.buffer_size = buffer_size
        self.object_buffer = [] # Object buffer stores information about tracked objects
        self.partial_object_buffer = [] # Object buffer stores information about the top, left and right halves and top left, top right corners of tracked objects
        self.object_in_frame = [] # Number of objects in each frame
        self.id = 0 # Current ID for assigning to new objects
        self.lambda_value = lambda_value
    
    def multi_match(self, dist_matrix, matched_set, current_set):
        matched_list = []
        new_set = current_set.copy()
        if dist_matrix.numel() != 0:
            for _ in range(len(dist_matrix)):
                max_dist, row, col = self.get_max(dist_matrix)
                if max_dist == -2 or max_dist < self.threshold:
                    break
                else:
                    key_1_list = list(matched_set.keys())
                    key_2_list = list(current_set.keys())


                    if key_1_list[row] not in matched_list:
                        matched_list.append(key_1_list[row])
                    else:
                        print(f'id {key_1_list[row]} is already matched.')

                    new_set[key_1_list[row]] = new_set.pop(key_2_list[col])
                    dist_matrix[row,:] = -2
                    dist_matrix[:,col] = -2


        return new_set, matched_list


        
    def match(self, obeject_embeddings, partial_embeddings, info_list, rerank=True):
        """
        Match current objects to existing objects in the object buffer or assign new IDs.

        Args:
        - object_embeddings (tensor): List of embeddings for the current objects
        - partial_embeddings (tensor): List of embeddings for the top, left and right halves and top left, top right corners of the current objects
        - info_list (list): List of information about the current objects
        - rerank (bool): Flag to specify whether to perform re-ranking

        Returns:
        - id_list (list): List of IDs assigned to the current objects
        - output_dist_mat (tensor): Distance matrix used for matching
        - output_partial_dist_mat (tensor): Distance matrix used for matching the top, left and right halves and top left, top right corners of the current objects
        """
        
        id_list = [-1] * len(obeject_embeddings)
        
        motion_tracklet = [[0] * 2] * len(obeject_embeddings)

        output_dist_mat = torch.tensor(999)
        output_partial_dist_mat = torch.tensor(999)

        # Record the number of objects in the current frame
        self.object_in_frame.append(len(obeject_embeddings))
        self.partial_object_in_frame.append(len(partial_embeddings))

        # Matching objects to existing objects in the object buffer
        if self.object_buffer and obeject_embeddings.numel() != 0:
            
            gallery_embedding = self.get_gallery_embedding()
            partial_gallery_embedding = self.get_partial_gallery_embedding()
            

            # Re-ranking the distance matrix if specified
            if rerank:
                q_g_dist = self.dot(obeject_embeddings, gallery_embedding)
                q_q_dist = self.dot(obeject_embeddings, obeject_embeddings)
                g_g_dist = self.dot(gallery_embedding, gallery_embedding)
                
                dist_matrix = self.re_ranking(q_q_dist, q_g_dist, g_g_dist, self.lambda_value)
                output_dist_mat = dist_matrix.clone().detach()

                q_g_partial_dist = self.dot(obeject_embeddings, partial_gallery_embedding)
                g_g_partial_dist = self.dot(partial_gallery_embedding, partial_gallery_embedding)

                partial_dist_mat = self.re_ranking(q_q_dist, q_g_partial_dist, g_g_partial_dist, self.lambda_value)
                output_partial_dist_mat = partial_dist_mat.clone().detach()

                selected_id = []
                for _ in range(len(obeject_embeddings)):
                    min_dist, row, col = self.get_min(dist_matrix)
                    if min_dist == 2 or min_dist > self.threshold:
                        break
                    else:
                        matched = 1
                        
                        if matched == 1 and self.object_buffer[col][2] not in selected_id:
                            id_list[row] = self.object_buffer[col][2]
                            selected_id.append(self.object_buffer[col][2])
                            dist_matrix[row,:] = 2
                            dist_matrix[:,col] = 2
                        else:
                            dist_matrix[row][col] = 2
                            _ -= 1


            # Directly match based on cosine similarity if not re-ranking
            else:
                dist_matrix = self.compute_distmatrix(obeject_embeddings)
                output_dist_mat = dist_matrix.clone().detach()

                partial_dist_mat = self.compute_partial_distmatrix(obeject_embeddings)
                output_partial_dist_mat = partial_dist_mat.clone().detach()

                selected_id = []
                for _ in range(len(obeject_embeddings)):
                    max_dist, row, col = self.get_max(dist_matrix)
                    if max_dist == -2 or max_dist < self.threshold:
                        break
                    else:
                        matched = 1
                        
                        if matched == 1 and self.object_buffer[col][2] not in selected_id:
                            id_list[row] = self.object_buffer[col][2]
                            selected_id.append(self.object_buffer[col][2])
                            dist_matrix[row,:] = -2
                            dist_matrix[:,col] = -2
                        else:
                            dist_matrix[row][col] = -2
                            _ -= 1

        # Assigning new IDs to unmatched objects
        for i in range(len(obeject_embeddings)):
            if id_list[i] == -1:
                id_list[i] = self.get_id(obeject_embeddings[i])
        
        # Add current objects into the object buffer
        for i in range(len(obeject_embeddings)):
            object_info = [obeject_embeddings[i].cpu().numpy(), info_list[i], id_list[i], motion_tracklet[i]]
            self.object_buffer.append(object_info)

        for i in range(len(partial_embeddings)):
            partial_object_info = [partial_embeddings[i].cpu().numpy(), info_list[i//5], id_list[i//5], motion_tracklet[i//5]]
            self.partial_object_buffer.append(partial_object_info)

        # Remove old objects from the object buffer if buffer size exceeds the limit
        if len(self.object_in_frame) > self.buffer_size:
            for i in range(self.object_in_frame[0]):
                self.object_buffer.pop(0)
            self.object_in_frame.pop(0)

        if len(self.partial_object_in_frame) > self.buffer_size:
            for i in range(self.partial_object_in_frame[0]):
                self.partial_object_buffer.pop(0)
            self.partial_object_in_frame.pop(0)
        
        return id_list, output_dist_mat, output_partial_dist_mat

    def get_min(self, dist_matrix):
        """
        Get the minimum value and its corresponding indices from the distance matrix.

        Args:
        - dist_matrix (tensor): Distance matrix between current and existing objects

        Returns:
        - min_dist (float): Minimum distance value
        - row (int): Index of the row containing the minimum value
        - col (int): Index of the column containing the minimum value
        """
        min_dist = dist_matrix.min()
        min_index = dist_matrix.argmin()
        row = (min_index // len(dist_matrix[0])).item() 
        col = (min_index % len(dist_matrix[0])).item()
        return min_dist, row, col

    def get_max(self, dist_matrix):

        """
        Get the maximum value and its corresponding indices from the distance matrix.

        Args:
        - dist_matrix (tensor): Distance matrix between current and existing objects

        Returns:
        - max_dist (float): Maximum distance value
        - row (int): Index of the row containing the maximum value
        - col (int): Index of the column containing the maximum value
        """

        max_dist = dist_matrix.max()
        max_index = dist_matrix.argmax()
        row = (max_index // len(dist_matrix[0])).item() 
        col = (max_index % len(dist_matrix[0])).item()
        return max_dist, row, col

    def get_gallery_embedding(self):
        """
        Get embeddings of objects in the gallery (object buffer).

        Returns:
        - gallery_embedding (np.array): Array of embeddings of objects in the gallery
        """
        gallery_embedding = []
        gallery_len = len(self.object_buffer)
        for i in range(gallery_len):
            gallery_embedding.append(self.object_buffer[i][0])

        return np.array(gallery_embedding)

    def get_partial_gallery_embedding(self):
        """
        Get embeddings of the top, left and right halves and top left, top right corners of objects in the gallery (partial object buffer).

        Returns:
        - partial_gallery_embedding (np.array): Array of embeddings of the top, left and right halves and top left, top right corners of objects in the gallery
        """
        partial_gallery_embedding = []
        gallery_len = len(self.partial_object_buffer)
        for i in range(gallery_len):
            partial_gallery_embedding.append(self.partial_object_buffer[i][1])

        return np.array(partial_gallery_embedding)


    
    def re_ranking(self, q_q_dist, q_g_dist, g_g_dist, lambda_value=0.3):
        """
        Re-rank the distance matrix using k-reciprocal nearest neighbors.

        Args:
        - q_q_dist (np.array): Distance matrix within query objects
        - q_g_dist (np.array): Distance matrix between query and gallery objects
        - g_g_dist (np.array): Distance matrix within gallery objects
        - lambda_value (float): Lambda value for balancing original and re-ranked distances

        Returns:
        - final_dist (np.array): Re-ranked distance matrix
        """

        # Set the default value of k
        k = 6

        # Concatenate distance matrices to form the initial distance matrix
        dist_matrix = np.concatenate(
            [np.concatenate([q_q_dist, q_g_dist], axis=1),
            np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
            axis=0)
        
        # Convert cosine distances to Mahalanobis distances
        dist_matrix = 2 - 2 * dist_matrix

        # Normalize the distance matrix
        dist_matrix = np.transpose(1. * dist_matrix/np.max(dist_matrix, axis = 0))
        
        # Adjust k if the number of objects is less than k+1
        if len(dist_matrix) < k+1:
            k = len(dist_matrix)-1
        
        # Initialize variables for k-reciprocal neighbors
        k2 = 1
        if k >= 2:
            k2 = k//2

        # Compute the initial rank based on the distance matrix
        initial_rank = np.argpartition(dist_matrix, range(1,k+1))
        
        # Initialize the initial vector V
        V = np.zeros_like(dist_matrix).astype(np.float32)
    
        # Get the number of query and gallery objects
        query_num = q_g_dist.shape[0]
        all_num = dist_matrix.shape[0]

        # Calculate k-reciprocal neighbors
        for i in range(all_num):
          
            forward_k_neigh_index = initial_rank[i, :k + 1]
            backward_k_neigh_index = initial_rank[forward_k_neigh_index, : k + 1]
            fi = np.where(backward_k_neigh_index == i)[0]
            k_reciprocal_index = forward_k_neigh_index[fi]
            k_reciprocal_expansion_index = k_reciprocal_index

            for j in range(len(k_reciprocal_index)):
                candidate = k_reciprocal_index[j]
                candidate_forward_k_neigh_index = initial_rank[candidate,
                                                :int(np.around(k / 2.)) + 1]
                candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                                :int(np.around(k / 2.)) + 1]
                fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
                candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
                if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                        candidate_k_reciprocal_index):
                    k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

            k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
            weight = np.exp(-dist_matrix[i, k_reciprocal_expansion_index])
            V[i, k_reciprocal_expansion_index] = 1. * weight / np.sum(weight)

        dist_matrix = dist_matrix[:query_num, ]

        if k2 != 1:
            V_qe = np.zeros_like(V, dtype=np.float32)
            for i in range(all_num):
                V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
            V = V_qe
            del V_qe
        del initial_rank

        invIndex = []

        for i in range(all_num):
            invIndex.append(np.where(V[:, i] != 0)[0])

        jaccard_dist = np.zeros_like(dist_matrix, dtype=np.float32)

        for i in range(query_num):
            temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
            indNonZero = np.where(V[i, :] != 0)[0]
            indImages = [invIndex[ind] for ind in indNonZero]
            for j in range(len(indNonZero)):
                temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                                V[indImages[j], indNonZero[j]])
            jaccard_dist[i] = 1 - temp_min / (2. - temp_min)

        final_dist = jaccard_dist * (1 - lambda_value) + dist_matrix * lambda_value
        del dist_matrix, V, jaccard_dist
        final_dist = final_dist[:query_num, query_num:]

        return final_dist
    
    def dot(self, x, y):
        """
        Compute dot product between two arrays.

        Args:
        - x (array): First array
        - y (array): Second array

        Returns:
        - dot product (float): Dot product between x and y
        """
        return np.dot(x, np.transpose(y))
    
    def compute_distmatrix(self, object_embeddings):

        """
        Compute the cosine similarity distance matrix between current and existing object embeddings.

        Args:
        - object_embeddings (tensor): Embeddings for current objects

        Returns:
        - dist_matrix (tensor): Distance matrix between current and existing objects
        """

        y_len = len(self.object_buffer)
        x_len= len(object_embeddings)
        dist_matrix = torch.empty((x_len, y_len))
        for i in range(x_len):
            for j in range(y_len):
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                dist_matrix[i][j] = torch.nn.functional.cosine_similarity(object_embeddings[i], torch.from_numpy(self.object_buffer[j][0]).to(device), dim=0)
        return dist_matrix


        def compute_partial_distmatrix(self, object_embeddings):

        """
        Compute the cosine similarity distance matrix between current object embeddings and existing partial object embeddings.

        Args:
        - object_embeddings (tensor): Embeddings for current objects

        Returns:
        - dist_matrix (tensor): Distance matrix between current objects and existing partial objects
        """

        y_len = len(self.partial_object_buffer)
        x_len= len(object_embeddings)
        dist_matrix = torch.empty((x_len, y_len))
        for i in range(x_len):
            for j in range(y_len):
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                dist_matrix[i][j] = torch.nn.functional.cosine_similarity(object_embeddings[i], torch.from_numpy(self.partial_object_buffer[j][0]).to(device), dim=0)
        return dist_matrix
    
    def get_id(self, obeject_embeddings):

        """
        Generate and return a new ID for an unmatched object.

        Args:
        - object_embeddings (tensor): Embeddings for the unmatched object

        Returns:
        - new_id (int): New ID for the unmatched object
        """
        
        self.id+=1
        return self.id-1


    def rule_1(self, x_motion, y_motion, x_offset, y_offset):   
        matched = 0
        motion = [0, 0]
        if x_motion == 0 and y_motion == 0:
 
            motion[0] = -1 if x_offset < 0 else 1
            motion[1] = -1 if y_offset < 0 else 1
            matched = 1
        else:
            x_offset -= ERROR * (1 if x_motion < 0 else -1)
            y_offset -= ERROR * (1 if y_motion < 0 else -1)
            if x_offset > 0 and y_offset > 0 and x_motion > 0 and y_motion > 0:
                motion = [1, 1]
                matched = 1
            elif x_offset < 0 and y_offset < 0 and x_motion < 0 and y_motion < 0:
                motion = [-1, -1]
                matched = 1
            elif x_offset > 0 and y_offset < 0 and x_motion > 0 and y_motion < 0:
                motion = [1, -1]
                matched = 1
            elif x_offset < 0 and y_offset > 0 and x_motion < 0 and y_motion > 0:
                motion = [-1, 1]
                matched = 1
        return matched, motion
    

    def nearBorder(self, info, cam):
        '''
        Return True if the object is near the border of the frame, False otherwise.

        Args:
        - info (list): Information about the object.

        Returns:
        - bool: True if the object is near the border, False otherwise.
        '''
        left_x = info[0]
        top_y = info[1]
        right_x = info[2]
        bottom_y = info[3]

        if cam == 0:
            if right_x <= 613 and bottom_y >= 710:
                return True
        elif cam == 1:
            if (right_x >= 1270 and top_y >= 406) or bottom_y >= 710:
                return True
        elif cam == 2:
            if right_x <= 596 and bottom_y >= 710:
                return True
        elif cam == 3:
            if left_x <= 10 and top_y >= 290:
                return True
        elif cam == 4:
            if (left_x <= 10 and top_y >= 312) or (right_x <= 624 and bottom_y >= 710):
                return True 
        elif cam == 5:
            if left_x >= 213 and bottom_y >= 710:
                return True
        elif cam == 6:
            return False
        elif cam == 7:
            if (left_x <= 10 and top_y >= 338) or (right_x <= 300 and bottom_y >= 710):
                return True

        return False




    
    def get_ensemble_id_list(self, object_embeddings, info_list, dist_matrices, cam, rerank=True):
        """
        Match current objects to existing objects in the object buffer or assign new IDs.

        Args:
        - object_embeddings (list): List of 360 frames of embeddings for the current objects.
        - info_list(list): List of information about the current objects.

        Returns:
        - id_list (list): List of IDs assigned to the current objects based on average of distance matrices from all models.
        """

        id_list = [-1] * len(object_embeddings)
        
        motion_tracklet = [[0] * 2] * len(object_embeddings)



        # Record the number of objects in the current frame
        self.object_in_frame.append(len(object_embeddings))
        self.partial_object_in_frame.append(len(object_embeddings)*5)

        # Matching objects to existing objects in the object buffer
        if self.object_buffer and object_embeddings.size != 0:
            
            #get the average distance matrix
            for i, matrix in enumerate(dist_matrices):
                if i == 0:
                    average_dist_matrix = matrix
                else:
                    average_dist_matrix += matrix

            average_dist_matrix /= len(dist_matrices)
           
            
            

            # Re-ranking the distance matrix if specified
            if rerank:
                selected_id = []
                for _ in range(len(object_embeddings)):
                    min_dist, row, col = self.get_min(average_dist_matrix)
                    if min_dist == 2 or min_dist > self.threshold:
                        break
                    else:
                        matched = 1
                        
                        if matched == 1 and self.object_buffer[col][2] not in selected_id:
                            id_list[row] = self.object_buffer[col][2]
                            selected_id.append(self.object_buffer[col][2])
                            average_dist_matrix[row,:] = 2
                            average_dist_matrix[:,col] = 2
                        else:
                            average_dist_matrix[row][col] = 2
                            _ -= 1
                # Check if there are any unmatched objects and try to match with partial objects
                if any(x == -1 for x in id_list) :
                    for idx in range(len(object_embeddings)):

                        if id_list[idx] != -1 or not(self.nearBorder(info_list[idx], cam)):

                            continue

                        for i, matrix in enumerate(partial_dist_matrices):
                            if i == 0:
                                average_partial_dist_matrix = matrix
                            else:
                                average_partial_dist_matrix += matrix
                        average_partial_dist_matrix /= len(partial_dist_matrices)

                        min_dist_partial, row_partial, col_partial = self.get_min(average_partial_dist_matrix)
                        if min_dist_partial == 2 or min_dist_partial > self.partial_threshold:
                            break
                        else:

                            matched = 1

                            if matched == 1 and self.partial_object_buffer[col_partial][2] not in selected_id:
                                selected_id.append(self.partial_object_buffer[col_partial][2])
                                id_list[row_partial//5] = self.partial_object_buffer[col_partial][2]

                                # mark ID for other parts of the same objects
                                if col_partial % 5 == 0: # top half selected
                                    average_partial_dist_matrix[row_partial,:] = 2
                                    average_partial_dist_matrix[:,col_partial] = 2

                                    average_partial_dist_matrix[:,col_partial+1] = 2
                                    average_partial_dist_matrix[:,col_partial+2] = 2
                                    average_partial_dist_matrix[:,col_partial+3] = 2
                                    average_partial_dist_matrix[:,col_partial+4] = 2

                                
                                elif col_partial % 5 == 1: # left half selected
                                    average_partial_dist_matrix[row_partial,:] = 2
                                    average_partial_dist_matrix[:,col_partial] = 2

                                    average_partial_dist_matrix[:,col_partial-1] = 2
                                    average_partial_dist_matrix[:,col_partial+1] = 2
                                    average_partial_dist_matrix[:,col_partial+2] = 2
                                    average_partial_dist_matrix[:,col_partial+3] = 2

                                
                                elif col_partial % 5 == 2: # right half selected
                                    average_partial_dist_matrix[row_partial,:] = 2
                                    average_partial_dist_matrix[:,col_partial] = 2

                                    average_partial_dist_matrix[:,col_partial-1] = 2
                                    average_partial_dist_matrix[:,col_partial-2] = 2
                                    average_partial_dist_matrix[:,col_partial+1] = 2
                                    average_partial_dist_matrix[:,col_partial+2] = 2
                                
                                elif col_partial % 5 == 3: # top left selected
                                    average_partial_dist_matrix[row_partial,:] = 2
                                    average_partial_dist_matrix[:,col_partial] = 2

                                    average_partial_dist_matrix[:,col_partial-1] = 2
                                    average_partial_dist_matrix[:,col_partial-2] = 2
                                    average_partial_dist_matrix[:,col_partial-3] = 2
                                    average_partial_dist_matrix[:,col_partial+1] = 2
                                
                                elif col_partial % 5 == 4: # top right selected
                                    average_partial_dist_matrix[row_partial,:] = 2
                                    average_partial_dist_matrix[:,col_partial] = 2

                                    average_partial_dist_matrix[:,col_partial-1] = 2
                                    average_partial_dist_matrix[:,col_partial-2] = 2
                                    average_partial_dist_matrix[:,col_partial-3] = 2
                                    average_partial_dist_matrix[:,col_partial-4] = 2


                                
                            else:
                                if col_partial % 5 == 0: # top half selected
                                    average_partial_dist_matrix[row_partial,:] = 2
                                    average_partial_dist_matrix[:,col_partial] = 2

                                    average_partial_dist_matrix[:,col_partial+1] = 2
                                    average_partial_dist_matrix[:,col_partial+2] = 2
                                    average_partial_dist_matrix[:,col_partial+3] = 2
                                    average_partial_dist_matrix[:,col_partial+4] = 2

                                
                                elif col_partial % 5 == 1: # left half selected
                                    average_partial_dist_matrix[row_partial,:] = 2
                                    average_partial_dist_matrix[:,col_partial] = 2

                                    average_partial_dist_matrix[:,col_partial-1] = 2
                                    average_partial_dist_matrix[:,col_partial+1] = 2
                                    average_partial_dist_matrix[:,col_partial+2] = 2
                                    average_partial_dist_matrix[:,col_partial+3] = 2

                                
                                elif col_partial % 5 == 2: # right half selected
                                    average_partial_dist_matrix[row_partial,:] = 2
                                    average_partial_dist_matrix[:,col_partial] = 2

                                    average_partial_dist_matrix[:,col_partial-1] = 2
                                    average_partial_dist_matrix[:,col_partial-2] = 2
                                    average_partial_dist_matrix[:,col_partial+1] = 2
                                    average_partial_dist_matrix[:,col_partial+2] = 2
                                
                                elif col_partial % 5 == 3: # top left selected
                                    average_partial_dist_matrix[row_partial,:] = 2
                                    average_partial_dist_matrix[:,col_partial] = 2

                                    average_partial_dist_matrix[:,col_partial-1] = 2
                                    average_partial_dist_matrix[:,col_partial-2] = 2
                                    average_partial_dist_matrix[:,col_partial-3] = 2
                                    average_partial_dist_matrix[:,col_partial+1] = 2
                                
                                elif col_partial % 5 == 4: # top right selected
                                    average_partial_dist_matrix[row_partial,:] = 2
                                    average_partial_dist_matrix[:,col_partial] = 2

                                    average_partial_dist_matrix[:,col_partial-1] = 2
                                    average_partial_dist_matrix[:,col_partial-2] = 2
                                    average_partial_dist_matrix[:,col_partial-3] = 2
                                    average_partial_dist_matrix[:,col_partial-4] = 2
                                idx -= 1


            # Directly match based on cosine similarity if not re-ranking
            else:
                selected_id = []
                for _ in range(len(object_embeddings)):
                    max_dist, row, col = self.get_max(average_dist_matrix)
                    if max_dist == -2 or max_dist < self.threshold:
                        break
                    else:
                        matched = 1
                        #try:
                        if matched == 1 and self.object_buffer[col][2] not in selected_id:
                            id_list[row] = self.object_buffer[col][2]
                            selected_id.append(self.object_buffer[col][2])
                            average_dist_matrix[row,:] = -2
                            average_dist_matrix[:,col] = -2
                        else:
                            average_dist_matrix[row][col] = -2
                            _ -= 1

                # Check if there are any unmatched objects and try to match with partial objects
                if any(x == -1 for x in id_list) :
                    for idx in range(len(object_embeddings)):

                        if id_list[idx] != -1 or not(self.nearBorder(info_list[idx], cam)):

                            continue

                        for i, matrix in enumerate(partial_dist_matrices):
                            if i == 0:
                                average_partial_dist_matrix = matrix
                            else:
                                average_partial_dist_matrix += matrix
                        average_partial_dist_matrix /= len(partial_dist_matrices)

                        max_dist_partial, row_partial, col_partial = self.get_max(average_partial_dist_matrix)
                        if max_dist_partial == -2 or max_dist_partial > self.partial_threshold:
                            break
                        else:

                            matched = 1

                            if matched == 1 and self.partial_object_buffer[col_partial][2] not in selected_id:
                                selected_id.append(self.partial_object_buffer[col_partial][2])
                                id_list[row_partial//5] = self.partial_object_buffer[col_partial][2]
                                
                                # mark ID for other parts of the same objects
                                if col_partial % 5 == 0: # top half selected
                                    average_partial_dist_matrix[row_partial,:] = -2
                                    average_partial_dist_matrix[:,col_partial] = -2

                                    average_partial_dist_matrix[:,col_partial+1] = -2
                                    average_partial_dist_matrix[:,col_partial+2] = -2
                                    average_partial_dist_matrix[:,col_partial+3] = -2
                                    average_partial_dist_matrix[:,col_partial+4] = -2

                                
                                elif col_partial % 5 == 1: # left half selected
                                    average_partial_dist_matrix[row_partial,:] = -2
                                    average_partial_dist_matrix[:,col_partial] = -2

                                    average_partial_dist_matrix[:,col_partial-1] = -2
                                    average_partial_dist_matrix[:,col_partial+1] = -2
                                    average_partial_dist_matrix[:,col_partial+2] = -2
                                    average_partial_dist_matrix[:,col_partial+3] = -2

                                
                                elif col_partial % 5 == 2: # right half selected
                                    average_partial_dist_matrix[row_partial,:] = -2
                                    average_partial_dist_matrix[:,col_partial] = -2

                                    average_partial_dist_matrix[:,col_partial-1] = -2
                                    average_partial_dist_matrix[:,col_partial-2] = -2
                                    average_partial_dist_matrix[:,col_partial+1] = -2
                                    average_partial_dist_matrix[:,col_partial+2] = -2
                                
                                elif col_partial % 5 == 3: # top left selected
                                    average_partial_dist_matrix[row_partial,:] = -2
                                    average_partial_dist_matrix[:,col_partial] = -2

                                    average_partial_dist_matrix[:,col_partial-1] = -2
                                    average_partial_dist_matrix[:,col_partial-2] = -2
                                    average_partial_dist_matrix[:,col_partial-3] = -2
                                    average_partial_dist_matrix[:,col_partial+1] = -2
                                
                                elif col_partial % 5 == 4: # top right selected
                                    average_partial_dist_matrix[row_partial,:] = -2
                                    average_partial_dist_matrix[:,col_partial] = -2

                                    average_partial_dist_matrix[:,col_partial-1] = -2
                                    average_partial_dist_matrix[:,col_partial-2] = -2
                                    average_partial_dist_matrix[:,col_partial-3] = -2
                                    average_partial_dist_matrix[:,col_partial-4] = -2


                                
                            else:
                                if col_partial % 5 == 0: # top half selected
                                    average_partial_dist_matrix[row_partial,:] = -2
                                    average_partial_dist_matrix[:,col_partial] = -2

                                    average_partial_dist_matrix[:,col_partial+1] = -2
                                    average_partial_dist_matrix[:,col_partial+2] = -2
                                    average_partial_dist_matrix[:,col_partial+3] = -2
                                    average_partial_dist_matrix[:,col_partial+4] = -2

                                
                                elif col_partial % 5 == 1: # left half selected
                                    average_partial_dist_matrix[row_partial,:] = -2
                                    average_partial_dist_matrix[:,col_partial] = -2

                                    average_partial_dist_matrix[:,col_partial-1] = -2
                                    average_partial_dist_matrix[:,col_partial+1] = -2
                                    average_partial_dist_matrix[:,col_partial+2] = -2
                                    average_partial_dist_matrix[:,col_partial+3] = -2

                                
                                elif col_partial % 5 == 2: # right half selected
                                    average_partial_dist_matrix[row_partial,:] = -2
                                    average_partial_dist_matrix[:,col_partial] = -2

                                    average_partial_dist_matrix[:,col_partial-1] = -2
                                    average_partial_dist_matrix[:,col_partial-2] = -2
                                    average_partial_dist_matrix[:,col_partial+1] = -2
                                    average_partial_dist_matrix[:,col_partial+2] = -2
                                
                                elif col_partial % 5 == 3: # top left selected
                                    average_partial_dist_matrix[row_partial,:] = -2
                                    average_partial_dist_matrix[:,col_partial] = -2

                                    average_partial_dist_matrix[:,col_partial-1] = -2
                                    average_partial_dist_matrix[:,col_partial-2] = -2
                                    average_partial_dist_matrix[:,col_partial-3] = -2
                                    average_partial_dist_matrix[:,col_partial+1] = -2
                                
                                elif col_partial % 5 == 4: # top right selected
                                    average_partial_dist_matrix[row_partial,:] = -2
                                    average_partial_dist_matrix[:,col_partial] = -2

                                    average_partial_dist_matrix[:,col_partial-1] = -2
                                    average_partial_dist_matrix[:,col_partial-2] = -2
                                    average_partial_dist_matrix[:,col_partial-3] = -2
                                    average_partial_dist_matrix[:,col_partial-4] = -2
                                idx -= 1


        # Assigning new IDs to unmatched objects
        for i in range(len(object_embeddings)):
            if id_list[i] == -1:
                id_list[i] = self.get_id(object_embeddings[i])
        
        # Add current objects into the object buffer
        for i in range(len(object_embeddings)):
            object_info = [object_embeddings[i], info_list[i], id_list[i], motion_tracklet[i]]
            self.object_buffer.append(object_info)

        for i in range(len(object_embeddings)*5):
            partial_object_info = [object_embeddings[i//5].cpu().numpy(), info_list[i//5], id_list[i//5], motion_tracklet[i//5]]
            self.partial_object_buffer.append(object_info)

        # Remove old objects from the object buffer if buffer size exceeds the limit
        if len(self.object_in_frame) > self.buffer_size:
            for i in range(self.object_in_frame[0]):
                self.object_buffer.pop(0)
            self.object_in_frame.pop(0)
        if len(self.partial_object_in_frame) > self.buffer_size:
            for i in range(self.partial_object_in_frame[0]):
                self.partial_object_buffer.pop(0)
            self.partial_object_in_frame.pop(0)

        return id_list