'''
Class to encode the Task's HTN representation, where each leaf node represents a concrete goal to be accomplished,
while internal nodes represent high level goals (compositions of lower level goals)
'''
import json
import os.path

class TaskHTN:

  def __init__(self, task_name=None, action_set=[], order = None):
    self._children = []
    self._values = {'name': task_name, 'actions': action_set, 'order_invariant': order, 'support_actions': []}

  def add_child(self,childHTN):
    self._children.append(childHTN)

  def remove_child(self,child):
    if child in self._children:
      self._children.remove(child)
      return True
    else:
      return False

  @property
  def children(self):
    return self._children

  @property
  def values(self):
    return self._values

  def is_order_invariant(self):
    return self._values['order_invariant'] is True

  def is_order_agnostic(self):
    return not self.is_order_invariant()

  def to_json(self):
    return json.dumps(self.to_dict())

  def to_dict(self):
    self_dict = {'values': self._values, 'children': []}

    if len(self._children) > 0:
      self_dict['children'] = [c.to_dict() for c in self._children]    

    return self_dict

  def get_path_to_node(target_node, path=[]):
    for c in self._children:
      next_step = get_path(c, path)
      if next_step is not None: 
        path.extend(next_step)
        break

    if target_node == self: 
      return [self]
    elif length(path) > 0:
      return path

    return None

  def save(self, filename, overwrite=False):
    if os.path.isfile(filename):
      if overwrite is False:
        return False
      print "Overwriting existing file..."
    f = open(filename, 'w')
    my_json = self.to_json()
    f.write(my_json)
    f.close()


  def load(filename):
    if os.path.isfile(filename) is False:
      return None

    htn_json = None
    with open(filename,'r') as f:
      htn_json = f.read()

    return TaskHTN.from_json(htn_json)

  def from_json(json):
    root = TaskHTN(None)
    root._values = json['values']
    for c in json['children']:
      root.add_child(TaskHTN.from_json(c))
    return root



class State:

  def __init__(self, feature_vector):
    self._features = feature_vector