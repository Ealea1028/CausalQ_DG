import torch
from causalq.losses.hpi_localization import patch_localization_loss
from causalq.models.hpi_query_route import QueryFeatureRouter


def test_soft_class_area_excludes_ignore_and_routes_gradient():
    scores=torch.tensor([[[2., 2.],[-1.,-1.]]],requires_grad=True)
    labels=torch.tensor([[[0,1,255,255],[1,255,255,255]]])
    loss=patch_localization_loss(scores,labels,(1,2))
    expected=-(scores[0,:,0].log_softmax(0)*torch.tensor([1/3,2/3])).sum()
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert torch.count_nonzero(scores.grad[:,:,1])==0
    router=QueryFeatureRouter(32,route_width=32)
    anchors=torch.randn(2,32,requires_grad=True)
    route=router(torch.randn(1,2,32),anchors,(1,2))
    patch_localization_loss(route.match_logits,labels,(1,2)).backward()
    assert anchors.grad.abs().sum()>0 and router.key.weight.grad.abs().sum()>0
