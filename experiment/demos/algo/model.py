from typing import List, Optional, Union

from pydantic import BaseModel, Field, RootModel


class MovieItem(BaseModel):
    title: str
    plot: str
    genres: str
    cover: str

class MetricData(BaseModel):
    dataset: str
    participation: int
    list_A: List[MovieItem]
    list_B: List[MovieItem]
    list_C: List[MovieItem]
    selected_list: str
    list_metrics: List[str]
    cf_ild: str
    cb_ild: str
    bin_div: str


class MagnitudeDatum(BaseModel):
    participation: int
    dataset: str
    compare_alphas_metric: str
    alphas: str
    approx_alphas: str
    gold: str
    selected: str
    list1: List[MovieItem]
    list2: List[MovieItem]
    list3: List[MovieItem]
    cf_ild: List[Union[int, str]]
    cb_ild: List[Union[int, str]]
    ease_ild: List[Union[int, str]]
    genres: List[Union[int, str]]
    tags: List[Union[int, str]]
    bin_div: List[Union[int, str]]


class Iteration(BaseModel):
    items: List[MovieItem]
    selected_items: List[MovieItem]
    cf_ild: float
    cb_ild: float
    bin_div: float

class Iterations(BaseModel):
    iter_0: Optional[Iteration] = Field(None, alias='0')
    iter_1: Optional[Iteration] = Field(None, alias='1')
    iter_2: Optional[Iteration] = Field(None, alias='2')
    iter_3: Optional[Iteration] = Field(None, alias='3')
    iter_4: Optional[Iteration] = Field(None, alias='4')
    iter_5: Optional[Iteration] = Field(None, alias='5')
    iter_6: Optional[Iteration] = Field(None, alias='6')
    iter_7: Optional[Iteration] = Field(None, alias='7')
    iter_8: Optional[Iteration] = Field(None, alias='8')
    iter_9: Optional[Iteration] = Field(None, alias='9')
    iter_10: Optional[Iteration] = Field(None, alias='10')
    iter_11: Optional[Iteration] = Field(None, alias='11')
    iter_12: Optional[Iteration] = Field(None, alias='12')
    iter_13: Optional[Iteration] = Field(None, alias='13')
    iter_14: Optional[Iteration] = Field(None, alias='14')
    iter_15: Optional[Iteration] = Field(None, alias='15')
    iter_16: Optional[Iteration] = Field(None, alias='16')
    iter_17: Optional[Iteration] = Field(None, alias='17')


class Iteration(BaseModel):
    block: int
    iterations: Iterations
    diversity_score: float
    serendipity_score: float
    dataset: str


class RecommendationData(BaseModel):
    participation: int
    elicitation_selections: List[MovieItem]
    iterations: List[Iteration]


class ModelItem(BaseModel):
    participation: int
    metric_data: MetricData
    magnitude_data: List[MagnitudeDatum]
    recommendation_data: RecommendationData


class Model(RootModel[List[ModelItem]]):
    pass
