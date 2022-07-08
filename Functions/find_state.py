def find_rowindex(S,row,col):
    row_index = -1
    state_space = S.shape[0]
    for i in range(state_space):
            if (S[i,0] == row) & (S[i,1] == col):
                row_index = i
    return(row_index)