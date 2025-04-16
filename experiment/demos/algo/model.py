from typing import List, Optional, Union, Dict

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

class Iteration(BaseModel):
    block: int
    iterations: Dict[str, Iteration]
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
