# -*- coding: utf-8 -*-
import os

# pip install tencentcloud-sdk-python
from tencentcloud.common import credential 
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException

# 导入对应产品模块的client models。
from tencentcloud.cvm.v20170312 import cvm_client, models

# 导入可选配置类
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile

class TencentAPI(object):
    def __init__(self):
        self._client = None
    
    def connect(self, region='shanghai'):
        try:
            # 实例化一个认证对象，入参需要传入腾讯云账户secretId，secretKey
            cred = credential.Credential(os.environ.get("TENCENTCLOUD_SECRET_ID"), os.environ.get("TENCENTCLOUD_SECRET_KEY"))

            # 实例化一个http选项，可选的，没有特殊需求可以跳过。
            httpProfile = HttpProfile()
            httpProfile.reqMethod = "GET"  # post请求(默认为post请求)
            httpProfile.reqTimeout = 30    # 请求超时时间，单位为秒(默认60秒)
            httpProfile.endpoint = "cvm.ap-%s.tencentcloudapi.com" % region # 指定接入地域域名(默认就近接入)

            # 实例化一个client选项，可选的，没有特殊需求可以跳过。
            clientProfile = ClientProfile()
            clientProfile.signMethod = "TC3-HMAC-SHA256"  # 指定签名算法
            clientProfile.language = "en-US"
            clientProfile.httpProfile = httpProfile

            # 实例化要请求产品(以cvm为例)的client对象，clientProfile是可选的。
            self._client = cvm_client.CvmClient(cred, "ap-%s" %region, clientProfile)
        except TencentCloudSDKException as err:
            print(err)    

        return self._client

    def listInstances(self, zones=["ap-shanghai-1", "ap-shanghai-2"]):
        try:
            # 实例化一个cvm实例信息查询请求对象,每个接口都会对应一个request对象。
            req = models.DescribeInstancesRequest()

            # 填充请求参数,这里request对象的成员变量即对应接口的入参。
            # 你可以通过官网接口文档或跳转到request对象的定义处查看请求参数的定义。
            respFilter = models.Filter()  # 创建Filter对象, 以zone的维度来查询cvm实例。
            respFilter.Name = "zone"
            respFilter.Values = zones
            req.Filters = [respFilter]  # Filters 是成员为Filter对象的列表

            # 这里还支持以标准json格式的string来赋值请求参数的方式。下面的代码跟上面的参数赋值是等效的。
            params = '{"Filters": [ { "Name": "zone", "Values": ["ap-shanghai-1", "ap-shanghai-2"] } ] }'
            req.from_json_string(params)

            # 通过client对象调用DescribeInstances方法发起请求。注意请求方法名与请求对象是对应的。
            # 返回的resp是一个DescribeInstancesResponse类的实例，与请求对象对应。
            print(req.to_json_string(indent=2))
            resp = self._client.DescribeInstances(req)

            # 输出json格式的字符串回包
            print(resp.to_json_string(indent=2))

            # 也可以取出单个值。
            # 你可以通过官网接口文档或跳转到response对象的定义处查看返回字段的定义。
            print(resp.TotalCount)

        except TencentCloudSDKException as err:
            print(err)

    def inquiryPrice(self, zone="ap-shanghai-2", bid =False, publicIp=True):
        try:
            # 实例化一个cvm实例信息查询请求对象,每个接口都会对应一个request对象。
            req = models.InquiryPriceRunInstancesRequest()
            req.InstanceCount=1
            req.InstanceType= 'GN6S.LARGE20' # 'S5.LARGE8'
            req.ImageId = 'img-dscg6q6b' # public 'img-9qabwvbn'

            req.Placement = models.Placement()
            req.Placement.Zone = zone
            req.VirtualPrivateCloud = models.VirtualPrivateCloud() #描述了VPC相关信息，包括子网/IP
            req.VirtualPrivateCloud.VpcId = 'DEFAULT' # 'vpc-lszzsaow'
            req.VirtualPrivateCloud.SubnetId = 'DEFAULT' # 私有网络子网ID，形如subnet-xxx。有效的私有网络子网ID可通过登录控制台查询；也可以调用接口 DescribeSubnets

            if bid:
                req.InstanceChargeType = 'SPOTPAID' # 实例计费类型: PREPAID：预付费/包年包月; POSTPAID_BY_HOUR：按小时后付费; SPOTPAID：竞价付费; 默认值：POSTPAID_BY_HOUR。

            req.InternetAccessible = models.InternetAccessible()
            if not publicIp:
                req.InternetAccessible.PublicIpAssigned = False
                req.InternetAccessible.InternetMaxBandwidthOut=0
            else:
                req.InternetAccessible.PublicIpAssigned = True
                req.InternetAccessible.InternetChargeType = 'BANDWIDTH_POSTPAID_BY_HOUR' # 网络计费类型。BANDWIDTH_PREPAID：预付费按带宽结算; TRAFFIC_POSTPAID_BY_HOUR：流量按小时后付费; BANDWIDTH_POSTPAID_BY_HOUR：带宽按小时后付费; BANDWIDTH_PACKAGE：带宽包用户. 默认取值：非带宽包用户默认与子机付费类型保持一致。
                req.InternetAccessible.InternetMaxBandwidthOut=7

            # req.SystemDisk = models.SystemDisk()
            # req.SystemDisk.DiskType='CLOUD_PREMIUM'
            # req.SystemDisk.DiskSize=50
            # req.InstanceName='QCLOUD-TEST'
            # req.LoginSettings.Password='Qcloud@TestApi123++'
            # req.EnhancedService.SecurityService.Enabled=TRUE
            # req.EnhancedService.MonitorService.Enabled=TRUE

            print(req.to_json_string(indent=2))

            resp = self._client.InquiryPriceRunInstances(req)
            # 输出json格式的字符串回包
            print(resp.to_json_string(indent=2))

        except TencentCloudSDKException as err:
            print(err)

    def createInstances(self, zone="ap-shanghai-2", bid =False, publicIp=True):
        try:
            # 实例化一个cvm实例信息查询请求对象,每个接口都会对应一个request对象。
            req = models.RunInstancesRequest()
            '''
                {
                "Placement": null,
                "ImageId": null,
                "InstanceChargeType": null,
                "InstanceChargePrepaid": null,
                "InstanceType": null,
                "SystemDisk": null,
                "DataDisks": null,
                "VirtualPrivateCloud": null,
                "InternetAccessible": null,
                "InstanceCount": null,
                "InstanceName": null,
                "LoginSettings": null,
                "SecurityGroupIds": null,
                "EnhancedService": null,
                "ClientToken": null,
                "HostName": null,
                "ActionTimer": null,
                "DisasterRecoverGroupIds": null,
                "TagSpecification": null,
                "InstanceMarketOptions": null,
                "UserData": null,
                "DryRun": null
                }            
            '''
            req.Placement = models.Placement()
            req.Placement.Zone = zone
            req.ImageId = 'img-9qabwvbn'

            print(req.to_json_string(indent=2))

            resp = self._client.RunInstances(req)
            # 输出json格式的字符串回包
            print(resp.to_json_string(indent=2))

        except TencentCloudSDKException as err:
            print(err)



api = TencentAPI()
api.connect()
# api.listInstances()
api.inquiryPrice()
# api.createInstances()