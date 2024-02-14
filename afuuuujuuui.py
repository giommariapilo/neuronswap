import torch


# tensor1 = torch.Tensor([[[[-0.0161, -0.0293,  0.0048,  0.0346,  0.0655],
#                            [-0.0117, -0.0193,  0.0188,  0.0464,  0.0757],
#                            [-0.0211, -0.0304,  0.0093,  0.0377,  0.0697],
#                            [-0.0321, -0.0398,  0.0038,  0.0306,  0.0583],
#                            [-0.0431, -0.0477, -0.0052,  0.0254,  0.0520]]],

#                          [[[ 0.0076, -0.0133, -0.0318, -0.0556, -0.0619],
#                            [ 0.0021, -0.0206, -0.0316, -0.0460, -0.0460],
#                            [-0.0296, -0.0505, -0.0533, -0.0629, -0.0595],
#                            [ 0.0205,  0.0040,  0.0017, -0.0077, -0.0142],
#                            [ 0.0008, -0.0136, -0.0119, -0.0346, -0.0469]]],

#                           [[[-0.0013,  0.0186,  0.0145,  0.0273,  0.0534],
#                            [-0.0054,  0.0114,  0.0105,  0.0249,  0.0519],
#                            [-0.0288, -0.0054,  0.0083,  0.0301,  0.0554],
#                            [-0.0332, -0.0027,  0.0166,  0.0409,  0.0595],
#                            [-0.0090,  0.0199,  0.0262,  0.0473,  0.0678]]],

#                          [[[ 0.0181,  0.0133,  0.0030,  0.0042,  0.0051],
#                            [ 0.0117,  0.0135,  0.0035, -0.0004,  0.0034],
#                            [ 0.0133,  0.0111,  0.0046,  0.0021,  0.0098],
#                            [ 0.0030, -0.0006, -0.0046, -0.0034,  0.0047],
#                            [-0.0132, -0.0105, -0.0107, -0.0132, -0.0054]]],

#                           [[[-0.0440, -0.0150, -0.0027, -0.0047, -0.0194],
#                            [-0.0054,  0.0081,  0.0246,  0.0198,  0.0013],
#                            [ 0.0109,  0.0251,  0.0335,  0.0325,  0.0133],
#                            [ 0.0082,  0.0160,  0.0266,  0.0231,  0.0073],
#                            [-0.0115, -0.0098,  0.0026,  0.0039, -0.0099]]],

#                          [[[ 0.0027, -0.0131,  0.0079,  0.0553,  0.0853],
#                            [ 0.0119, -0.0090,  0.0049,  0.0521,  0.0845],
#                            [ 0.0045, -0.0228, -0.0106,  0.0395,  0.0701],
#                            [-0.0115, -0.0364, -0.0271,  0.0222,  0.0508],
#                            [-0.0228, -0.0444, -0.0372,  0.0065,  0.0337]]]])
tensor1 = torch.Tensor([-1.0001e-06,  1.4361e-07,  2.1791e-07,  7.8905e-08,  3.2799e-08, -2.9274e-07])

tensor2 = torch.Tensor([[0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1]])

tensor2 = tensor2.transpose(1, 0)
print(tensor2)
weight_shape = tensor1.shape
if len(weight_shape) != len(tensor2.shape):
  reshaped_weight = torch.reshape(tensor1, (weight_shape[0], -1))
  permutated_weight = torch.matmul(tensor2, reshaped_weight)
  tensor1 = torch.reshape(permutated_weight, weight_shape)
else: 
  tensor1 = torch.matmul(tensor2, tensor1)

print(tensor1)