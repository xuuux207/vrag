"""
企业图谱数据 - 用于实验2 RAG融合测试

虚构企业生态系统，围绕 TechFlow 工业解决方案展开
目的：
1. 测试企业图谱 + 业务文档的混合 RAG 检索
2. 验证 BM25 + Dense 混合检索的必要性
3. 避免训练数据污染，真实测试模型推理能力

设计原则：
- 所有企业、人名、数据完全虚构
- 与实验一的 TechFlow 产品线对齐
- 包含成功案例、竞品客户、潜在客户等多种状态
- 添加 synthetic_source 水印标记
"""

from typing import List, Dict
from datetime import datetime

# ========== 企业图谱主数据 ==========

COMPANY_GRAPH: List[Dict] = [
    {
        "id": "company_001",
        "name": "鼎盛科技有限公司",
        "industry": "智能制造",
        "founded_date": "2015-03-15",
        "capital": "5000万元",
        "business_scope": "工业自动化、数据采集、IoT平台",
        "employees": 500,
        "ceo": "李明",
        "parent_company": "鼎盛集团",
        "subsidiaries": ["鼎盛智能", "鼎盛数据"],
        "investors": ["红杉资本", "IDG"],
        "past_projects": [
            {
                "year": 2023,
                "project": "工厂数字化改造",
                "partner": "TechFlow",
                "product": "DataStream Lite",
                "result": "生产效率提升 35%",
                "investment": "480万元",
                "duration": "8周"
            }
        ],
        "pain_points": ["生产数据孤岛", "设备故障率高", "库存管理混乱"],
        "synthetic_source": "experiment_v2",
        "created_date": "2025-12-14"
    },
    {
        "id": "company_002",
        "name": "星辰金融集团",
        "industry": "金融科技",
        "founded_date": "2010-06-01",
        "capital": "2亿元",
        "business_scope": "证券交易、风控系统、支付平台",
        "employees": 2000,
        "ceo": "王芳",
        "parent_company": None,
        "subsidiaries": ["星辰证券", "星辰支付", "星辰保险"],
        "investors": ["腾讯", "软银"],
        "past_projects": [],  # 未合作过
        "pain_points": ["实时风控延迟高", "交易数据处理瓶颈", "合规压力大"],
        "note": "潜在客户，正在评估 TechFlow 方案",
        "synthetic_source": "experiment_v2",
        "created_date": "2025-12-14"
    },
    {
        "id": "company_003",
        "name": "明远物流科技",
        "industry": "智慧物流",
        "founded_date": "2018-09-20",
        "capital": "8000万元",
        "business_scope": "仓储管理、智能调度、物流追踪",
        "employees": 800,
        "ceo": "赵强",
        "parent_company": "明远集团",
        "subsidiaries": ["明远仓储", "明远配送"],
        "investors": ["京东", "菜鸟网络"],
        "past_projects": [
            {
                "year": 2022,
                "project": "智慧仓储系统",
                "partner": "TechFlow",
                "product": "FlowControl-X100",
                "result": "仓储效率提升 28%，拣货错误率降低 45%",
                "investment": "120万元",
                "duration": "6周"
            }
        ],
        "pain_points": ["仓储空间利用率低", "人工拣货错误率高", "运输路线优化困难"],
        "synthetic_source": "experiment_v2",
        "created_date": "2025-12-14"
    },
    {
        "id": "company_004",
        "name": "云帆新能源",
        "industry": "新能源电池",
        "founded_date": "2016-11-08",
        "capital": "3亿元",
        "business_scope": "锂电池研发、动力电池生产、储能系统",
        "employees": 1200,
        "ceo": "孙梅",
        "parent_company": None,
        "subsidiaries": ["云帆电池", "云帆储能"],
        "investors": ["宁德时代", "蔚来资本"],
        "past_projects": [],  # 未合作过(潜在客户)
        "pain_points": ["电池一致性差", "生产线过热停机频繁", "质量追溯困难"],
        "note": "高潜力客户，痛点与 TechFlow 产品高度匹配",
        "synthetic_source": "experiment_v2",
        "created_date": "2025-12-14"
    },
    {
        "id": "company_005",
        "name": "光明医疗器械",
        "industry": "医疗设备",
        "founded_date": "2012-04-15",
        "capital": "1.5亿元",
        "business_scope": "医疗器械制造、质量认证、设备维护",
        "employees": 600,
        "ceo": "钱伟",
        "parent_company": "光明健康集团",
        "subsidiaries": ["光明影像", "光明检验"],
        "investors": ["红杉资本", "高瓴资本"],
        "past_projects": [
            {
                "year": 2024,
                "project": "GMP质量追溯系统",
                "partner": "TechFlow",
                "product": "FlowMind 平台",
                "result": "FDA审计一次通过，追溯效率提升 90%",
                "investment": "200万元",
                "duration": "10周"
            }
        ],
        "pain_points": ["GMP合规压力大", "批次追溯复杂", "设备校准管理困难"],
        "synthetic_source": "experiment_v2",
        "created_date": "2025-12-14"
    },
    {
        "id": "company_006",
        "name": "恒通电子商务",
        "industry": "电商平台",
        "founded_date": "2014-07-01",
        "capital": "5亿元",
        "business_scope": "B2C电商、供应链管理、数据分析",
        "employees": 3000,
        "ceo": "周丽",
        "parent_company": None,
        "subsidiaries": ["恒通商城", "恒通支付", "恒通云仓"],
        "investors": ["阿里巴巴", "IDG"],
        "past_projects": [],  # 使用竞品
        "pain_points": ["库存周转慢", "供应链可视化差", "大促期间系统压力大"],
        "competitor_product": "XX竞品实时数据平台",
        "note": "竞品客户，需谨慎推荐并说明差异化优势",
        "synthetic_source": "experiment_v2",
        "created_date": "2025-12-14"
    },
    {
        "id": "company_007",
        "name": "天启化工集团",
        "industry": "精细化工",
        "founded_date": "2008-03-22",
        "capital": "10亿元",
        "business_scope": "化工原料生产、工艺研发、环保治理",
        "employees": 2500,
        "ceo": "吴刚",
        "parent_company": None,
        "subsidiaries": ["天启材料", "天启环保", "天启研究院"],
        "investors": ["中石化", "中化集团"],
        "past_projects": [
            {
                "year": 2023,
                "project": "化工生产安全监控",
                "partner": "TechFlow",
                "product": "FlowControl-X500",
                "result": "安全事故零发生，能耗降低 22%",
                "investment": "680万元",
                "duration": "12周"
            }
        ],
        "pain_points": ["工艺参数控制精度低", "安全风险高", "能耗成本居高不下"],
        "synthetic_source": "experiment_v2",
        "created_date": "2025-12-14"
    },
    {
        "id": "company_008",
        "name": "晨曦食品加工",
        "industry": "食品制造",
        "founded_date": "2010-05-18",
        "capital": "6000万元",
        "business_scope": "食品加工、冷链物流、品质检测",
        "employees": 400,
        "ceo": "郑敏",
        "parent_company": "晨曦集团",
        "subsidiaries": ["晨曦食品", "晨曦冷链"],
        "investors": ["中粮集团"],
        "past_projects": [
            {
                "year": 2021,
                "project": "食品安全追溯系统",
                "partner": "TechFlow",
                "product": "FlowControl-X100",
                "result": "追溯覆盖率100%，召回响应时间从3天缩短至2小时",
                "investment": "80万元",
                "duration": "5周"
            }
        ],
        "pain_points": ["食品安全追溯要求严格", "冷链温度监控不稳定", "生产批次管理混乱"],
        "synthetic_source": "experiment_v2",
        "created_date": "2025-12-14"
    },
    {
        "id": "company_009",
        "name": "智信半导体",
        "industry": "半导体制造",
        "founded_date": "2019-01-10",
        "capital": "20亿元",
        "business_scope": "芯片设计、晶圆制造、封装测试",
        "employees": 5000,
        "ceo": "冯涛",
        "parent_company": None,
        "subsidiaries": ["智信设计", "智信制造", "智信封测"],
        "investors": ["国家集成电路产业基金", "华为", "小米"],
        "past_projects": [],  # 潜在大客户
        "pain_points": ["良率波动大", "设备维护成本高", "生产数据孤岛严重"],
        "note": "高价值潜在客户，需定制化方案",
        "synthetic_source": "experiment_v2",
        "created_date": "2025-12-14"
    },
    {
        "id": "company_010",
        "name": "泰和纺织集团",
        "industry": "纺织服装",
        "founded_date": "2005-08-30",
        "capital": "4亿元",
        "business_scope": "面料生产、成衣制造、供应链管理",
        "employees": 1500,
        "ceo": "胡建国",
        "parent_company": None,
        "subsidiaries": ["泰和面料", "泰和服装", "泰和贸易"],
        "investors": ["申洲国际"],
        "past_projects": [
            {
                "year": 2022,
                "project": "智能染色工艺优化",
                "partner": "TechFlow",
                "product": "FlowMind 平台",
                "result": "染色一次合格率从75%提升至92%，节水25%",
                "investment": "150万元",
                "duration": "8周"
            }
        ],
        "pain_points": ["染色工艺一致性差", "能耗水耗高", "订单交期压力大"],
        "synthetic_source": "experiment_v2",
        "created_date": "2025-12-14"
    }
]

# ========== 企业关系网络 ==========

COMPANY_RELATIONS: List[Dict] = [
    # ========== 股权关系 ==========
    {"source": "鼎盛集团", "relation": "持股", "target": "鼎盛科技有限公司", "percentage": 80, "relation_type": "equity"},
    {"source": "鼎盛科技有限公司", "relation": "全资子公司", "target": "鼎盛智能", "percentage": 100, "relation_type": "equity"},
    {"source": "鼎盛科技有限公司", "relation": "全资子公司", "target": "鼎盛数据", "percentage": 100, "relation_type": "equity"},

    {"source": "明远集团", "relation": "持股", "target": "明远物流科技", "percentage": 60, "relation_type": "equity"},
    {"source": "明远物流科技", "relation": "控股子公司", "target": "明远仓储", "percentage": 75, "relation_type": "equity"},
    {"source": "明远物流科技", "relation": "控股子公司", "target": "明远配送", "percentage": 70, "relation_type": "equity"},

    {"source": "光明健康集团", "relation": "持股", "target": "光明医疗器械", "percentage": 90, "relation_type": "equity"},
    {"source": "光明医疗器械", "relation": "全资子公司", "target": "光明影像", "percentage": 100, "relation_type": "equity"},
    {"source": "光明医疗器械", "relation": "全资子公司", "target": "光明检验", "percentage": 100, "relation_type": "equity"},

    {"source": "晨曦集团", "relation": "持股", "target": "晨曦食品加工", "percentage": 85, "relation_type": "equity"},
    {"source": "晨曦食品加工", "relation": "全资子公司", "target": "晨曦食品", "percentage": 100, "relation_type": "equity"},
    {"source": "晨曦食品加工", "relation": "全资子公司", "target": "晨曦冷链", "percentage": 100, "relation_type": "equity"},

    # ========== 投资关系 ==========
    {"source": "红杉资本", "relation": "投资", "target": "鼎盛科技有限公司", "round": "B轮", "amount": "2亿元", "year": 2018, "relation_type": "investment"},
    {"source": "IDG", "relation": "投资", "target": "鼎盛科技有限公司", "round": "A轮", "amount": "5000万元", "year": 2016, "relation_type": "investment"},
    {"source": "红杉资本", "relation": "投资", "target": "光明医疗器械", "round": "C轮", "amount": "3亿元", "year": 2019, "relation_type": "investment"},
    {"source": "高瓴资本", "relation": "投资", "target": "光明医疗器械", "round": "C轮", "amount": "2亿元", "year": 2019, "relation_type": "investment"},

    {"source": "腾讯", "relation": "投资", "target": "星辰金融集团", "round": "战略投资", "amount": "10亿元", "year": 2015, "relation_type": "investment"},
    {"source": "软银", "relation": "投资", "target": "星辰金融集团", "round": "B轮", "amount": "5亿元", "year": 2012, "relation_type": "investment"},

    {"source": "京东", "relation": "投资", "target": "明远物流科技", "round": "B轮", "amount": "1.5亿元", "year": 2020, "relation_type": "investment"},
    {"source": "菜鸟网络", "relation": "投资", "target": "明远物流科技", "round": "B轮", "amount": "1亿元", "year": 2020, "relation_type": "investment"},

    {"source": "宁德时代", "relation": "投资", "target": "云帆新能源", "round": "A轮", "amount": "2亿元", "year": 2018, "relation_type": "investment"},
    {"source": "蔚来资本", "relation": "投资", "target": "云帆新能源", "round": "A轮", "amount": "1亿元", "year": 2018, "relation_type": "investment"},

    {"source": "阿里巴巴", "relation": "投资", "target": "恒通电子商务", "round": "C轮", "amount": "8亿元", "year": 2017, "relation_type": "investment"},
    {"source": "IDG", "relation": "投资", "target": "恒通电子商务", "round": "B轮", "amount": "3亿元", "year": 2016, "relation_type": "investment"},

    {"source": "中石化", "relation": "战略投资", "target": "天启化工集团", "round": "战略投资", "amount": "15亿元", "year": 2010, "relation_type": "investment"},
    {"source": "中化集团", "relation": "战略投资", "target": "天启化工集团", "round": "战略投资", "amount": "10亿元", "year": 2010, "relation_type": "investment"},

    {"source": "中粮集团", "relation": "战略投资", "target": "晨曦食品加工", "round": "战略投资", "amount": "1亿元", "year": 2015, "relation_type": "investment"},

    {"source": "国家集成电路产业基金", "relation": "投资", "target": "智信半导体", "round": "A轮", "amount": "30亿元", "year": 2020, "relation_type": "investment"},
    {"source": "华为", "relation": "战略投资", "target": "智信半导体", "round": "A轮", "amount": "5亿元", "year": 2020, "relation_type": "investment"},
    {"source": "小米", "relation": "战略投资", "target": "智信半导体", "round": "A轮", "amount": "3亿元", "year": 2020, "relation_type": "investment"},

    {"source": "申洲国际", "relation": "战略投资", "target": "泰和纺织集团", "round": "战略投资", "amount": "2亿元", "year": 2010, "relation_type": "investment"},

    # ========== TechFlow 合作关系（成功案例）==========
    {"source": "鼎盛科技有限公司", "relation": "使用产品", "target": "DataStream Lite", "year": 2023, "satisfaction": 4.5, "relation_type": "cooperation"},
    {"source": "TechFlow", "relation": "服务客户", "target": "鼎盛科技有限公司", "project": "工厂数字化改造", "year": 2023, "relation_type": "cooperation"},

    {"source": "明远物流科技", "relation": "使用产品", "target": "FlowControl-X100", "year": 2022, "satisfaction": 4.3, "relation_type": "cooperation"},
    {"source": "TechFlow", "relation": "服务客户", "target": "明远物流科技", "project": "智慧仓储系统", "year": 2022, "relation_type": "cooperation"},

    {"source": "光明医疗器械", "relation": "使用产品", "target": "FlowMind 平台", "year": 2024, "satisfaction": 4.8, "relation_type": "cooperation"},
    {"source": "TechFlow", "relation": "服务客户", "target": "光明医疗器械", "project": "GMP质量追溯系统", "year": 2024, "relation_type": "cooperation"},

    {"source": "天启化工集团", "relation": "使用产品", "target": "FlowControl-X500", "year": 2023, "satisfaction": 4.6, "relation_type": "cooperation"},
    {"source": "TechFlow", "relation": "服务客户", "target": "天启化工集团", "project": "化工生产安全监控", "year": 2023, "relation_type": "cooperation"},

    {"source": "晨曦食品加工", "relation": "使用产品", "target": "FlowControl-X100", "year": 2021, "satisfaction": 4.4, "relation_type": "cooperation"},
    {"source": "TechFlow", "relation": "服务客户", "target": "晨曦食品加工", "project": "食品安全追溯系统", "year": 2021, "relation_type": "cooperation"},

    {"source": "泰和纺织集团", "relation": "使用产品", "target": "FlowMind 平台", "year": 2022, "satisfaction": 4.5, "relation_type": "cooperation"},
    {"source": "TechFlow", "relation": "服务客户", "target": "泰和纺织集团", "project": "智能染色工艺优化", "year": 2022, "relation_type": "cooperation"},

    # ========== 竞品客户关系（陷阱测试数据）==========
    {"source": "恒通电子商务", "relation": "使用竞品", "target": "XX竞品实时数据平台", "year": 2021, "vendor": "XX竞品公司", "relation_type": "competitor"},
    {"source": "星辰金融集团", "relation": "潜在客户", "target": "TechFlow", "status": "未合作", "note": "评估中", "relation_type": "potential"},

    # ========== 潜在客户（未合作）==========
    {"source": "云帆新能源", "relation": "潜在客户", "target": "TechFlow", "status": "未合作", "industry": "新能源电池", "relation_type": "potential"},
    {"source": "智信半导体", "relation": "潜在客户", "target": "TechFlow", "status": "未合作", "industry": "半导体制造", "relation_type": "potential"},

    # ========== 供应链关系 ==========
    {"source": "鼎盛科技有限公司", "relation": "云服务商", "target": "华为云", "service": "IaaS云计算", "relation_type": "supply_chain"},
    {"source": "星辰金融集团", "relation": "云服务商", "target": "阿里云", "service": "金融云", "relation_type": "supply_chain"},
    {"source": "明远物流科技", "relation": "设备供应商", "target": "海康威视", "service": "仓储监控设备", "relation_type": "supply_chain"},
    {"source": "云帆新能源", "relation": "原材料供应商", "target": "赣锋锂业", "service": "锂材料", "relation_type": "supply_chain"},
    {"source": "光明医疗器械", "relation": "认证机构", "target": "SGS", "service": "ISO认证", "relation_type": "supply_chain"},
    {"source": "天启化工集团", "relation": "环保合作商", "target": "碧水源", "service": "废水处理", "relation_type": "supply_chain"},

    # ========== 行业竞争关系 ==========
    {"source": "云帆新能源", "relation": "同行竞争", "target": "比亚迪电池", "industry": "动力电池", "relation_type": "competition"},
    {"source": "智信半导体", "relation": "同行竞争", "target": "中芯国际", "industry": "晶圆代工", "relation_type": "competition"},
    {"source": "恒通电子商务", "relation": "同行竞争", "target": "京东商城", "industry": "B2C电商", "relation_type": "competition"},
]

# ========== 数据转换函数 ==========

def company_to_documents(company: Dict) -> List[Dict]:
    """
    将企业图谱转为多个检索文档

    每家企业生成 4 类文档：
    1. 企业基本信息
    2. 企业关系
    3. 合作历史（如有）
    4. 痛点分析
    """
    docs = []

    # 文档 1：企业基本信息
    docs.append({
        "id": f"{company['id']}_basic",
        "type": "company_profile",
        "source_type": "graph",
        "title": f"{company['name']} 企业档案",
        "content": f"""【企业名称】{company['name']}
【行业】{company['industry']}
【成立时间】{company['founded_date']}
【注册资本】{company['capital']}
【员工规模】{company['employees']}人
【CEO】{company['ceo']}
【经营范围】{company['business_scope']}
【母公司】{company.get('parent_company', '无')}
""",
        "metadata": {
            "company_id": company['id'],
            "company_name": company['name'],
            "industry": company['industry'],
            "synthetic_source": company['synthetic_source']
        }
    })

    # 文档 2：企业关系
    docs.append({
        "id": f"{company['id']}_relations",
        "type": "company_relations",
        "source_type": "graph",
        "title": f"{company['name']} 企业关系",
        "content": f"""【企业名称】{company['name']}
【母公司】{company.get('parent_company', '无')}
【子公司】{', '.join(company.get('subsidiaries', []))}
【投资方】{', '.join(company.get('investors', []))}
""",
        "metadata": {
            "company_id": company['id'],
            "company_name": company['name'],
            "synthetic_source": company['synthetic_source']
        }
    })

    # 文档 3：合作历史（如有）
    if company.get('past_projects'):
        for project in company['past_projects']:
            docs.append({
                "id": f"{company['id']}_project_{project['year']}",
                "type": "project_history",
                "source_type": "graph",
                "title": f"{company['name']} 合作案例（{project['year']}）",
                "content": f"""【客户】{company['name']}
【行业】{company['industry']}
【项目名称】{project['project']}
【时间】{project['year']}年
【合作方】{project['partner']}
【使用产品】{project['product']}
【项目成果】{project['result']}
【投资金额】{project.get('investment', '未公开')}
【实施周期】{project.get('duration', '未公开')}
""",
                "metadata": {
                    "company_id": company['id'],
                    "company_name": company['name'],
                    "project_year": project['year'],
                    "product": project['product'],
                    "synthetic_source": company['synthetic_source']
                }
            })

    # 文档 4：痛点分析
    docs.append({
        "id": f"{company['id']}_painpoints",
        "type": "company_needs",
        "source_type": "graph",
        "title": f"{company['name']} 业务痛点",
        "content": f"""【企业名称】{company['name']}
【行业】{company['industry']}
【员工规模】{company['employees']}人
【当前痛点】
{chr(10).join(f"- {pain}" for pain in company['pain_points'])}
{f"【备注】{company['note']}" if company.get('note') else ""}
""",
        "metadata": {
            "company_id": company['id'],
            "company_name": company['name'],
            "industry": company['industry'],
            "has_competitor_product": "competitor_product" in company,
            "synthetic_source": company['synthetic_source']
        }
    })

    return docs


def get_all_company_documents() -> List[Dict]:
    """获取所有企业的检索文档"""
    all_docs = []
    for company in COMPANY_GRAPH:
        all_docs.extend(company_to_documents(company))
    return all_docs


def get_companies_by_status(status: str) -> List[Dict]:
    """
    根据合作状态筛选企业

    Args:
        status: "cooperated" | "competitor" | "potential" | "all"
    """
    if status == "all":
        return COMPANY_GRAPH
    elif status == "cooperated":
        return [c for c in COMPANY_GRAPH if c.get('past_projects')]
    elif status == "competitor":
        return [c for c in COMPANY_GRAPH if c.get('competitor_product')]
    elif status == "potential":
        return [c for c in COMPANY_GRAPH if not c.get('past_projects') and not c.get('competitor_product')]
    else:
        return []


def get_company_by_name(name: str) -> Dict:
    """根据名称获取企业信息"""
    for company in COMPANY_GRAPH:
        if company['name'] == name:
            return company
    return None


def get_relations_by_company(company_name: str) -> List[Dict]:
    """获取某企业的所有关系"""
    relations = []
    for rel in COMPANY_RELATIONS:
        if rel['source'] == company_name or rel.get('target') == company_name:
            relations.append(rel)
    return relations


def get_relations_by_type(relation_type: str) -> List[Dict]:
    """
    根据关系类型筛选

    Args:
        relation_type: "equity" | "investment" | "cooperation" | "competitor" | "potential" | "supply_chain" | "competition"
    """
    return [r for r in COMPANY_RELATIONS if r.get('relation_type') == relation_type]


# ========== 数据统计 ==========

def get_statistics() -> Dict:
    """获取数据统计信息"""
    return {
        "total_companies": len(COMPANY_GRAPH),
        "total_relations": len(COMPANY_RELATIONS),
        "cooperated_companies": len(get_companies_by_status("cooperated")),
        "competitor_companies": len(get_companies_by_status("competitor")),
        "potential_companies": len(get_companies_by_status("potential")),
        "total_documents": len(get_all_company_documents()),
        "relation_types": {
            "equity": len(get_relations_by_type("equity")),
            "investment": len(get_relations_by_type("investment")),
            "cooperation": len(get_relations_by_type("cooperation")),
            "competitor": len(get_relations_by_type("competitor")),
            "potential": len(get_relations_by_type("potential")),
            "supply_chain": len(get_relations_by_type("supply_chain")),
            "competition": len(get_relations_by_type("competition")),
        },
        "industries": list(set(c['industry'] for c in COMPANY_GRAPH)),
        "created_date": "2025-12-14"
    }


if __name__ == "__main__":
    # 打印统计信息
    import json
    stats = get_statistics()
    print("=" * 60)
    print("企业图谱数据统计")
    print("=" * 60)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("\n企业列表:")
    for i, company in enumerate(COMPANY_GRAPH, 1):
        status = "已合作" if company.get('past_projects') else \
                 "竞品客户" if company.get('competitor_product') else "潜在客户"
        print(f"{i}. {company['name']:<15} | {company['industry']:<10} | {status}")
    print("=" * 60)
