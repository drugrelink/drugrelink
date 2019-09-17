# -*- coding: utf-8 -*-

"""Forms encoded in WTForms."""

from flask_wtf import FlaskForm
from wtforms.fields import StringField, SubmitField
from wtforms.validators import DataRequired

__all__ = [
    'QueryForm',
]


class QueryForm(FlaskForm):
    """Builds the form for querying the model."""

    curie = StringField('Entity', validators=[DataRequired()])
    submit_subgraph = SubmitField('Submit')
